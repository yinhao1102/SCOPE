"""Compare cached ORBench evaluation against the original 32-loader protocol."""

from collections import Counter
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from SCOPE.datasets.orbench.bases import GalleryDataset
from SCOPE.datasets.orbench.build import _create_query_datasets
from SCOPE.utils.cached_orbench_eval import (
    build_mean_feature_cache, mean_query_features, rank_feature_chunks,
    supports_metadata_cache,
)
from SCOPE.utils.evaluation import ORBENCH_QUERY_LOADERS
from SCOPE.utils.reid5o_metrics import Evaluator1, rank


class CountingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.counts = Counter()
        generator = torch.Generator().manual_seed(19)
        self.register_buffer("vision_projection", torch.randn(48, 13, generator=generator))
        self.register_buffer("text_projection", torch.randn(77, 13, generator=generator))

    def _encode(self, modality, values):
        self.counts[modality] += len(values)
        if modality == "text":
            return values.float() @ self.text_projection / 1000
        scale = {"rgb": 1, "nir": 2, "cp": 0.5, "sk": 3}[modality]
        return values.flatten(1) @ self.vision_projection * scale

    def encode_rgb_cls(self, values):
        return self._encode("rgb", values)

    def encode_nir_cls(self, values):
        return self._encode("nir", values)

    def encode_cp_cls(self, values):
        return self._encode("cp", values)

    def encode_sk_cls(self, values):
        return self._encode("sk", values)

    def encode_text_cls(self, values):
        return self._encode("text", values)


class CachedORBenchEvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        random = np.random.default_rng(3)
        self.paths = []
        for index in range(12):
            path = Path(self.temporary.name) / f"image_{index}.png"
            Image.fromarray(random.integers(0, 256, (4, 4, 3), dtype=np.uint8)).save(path)
            self.paths.append(str(path))
        self.transform = T.ToTensor()
        gallery = GalleryDataset([index % 3 for index in range(12)], self.paths, self.transform)
        self.gallery_loader = DataLoader(gallery, batch_size=3)
        empty = Evaluator1(self.gallery_loader)
        captions = ["person in black", " person in black ", "person in red", "person in blue"]
        query_rows = {}
        for name, modalities, _ in empty._get_modality_combinations():
            count = {"nir": 7, "cp": 6, "sk": 5, "text": 8}[modalities[0]]
            rows = []
            for index in range(count):
                values = [captions[(index + offset) % len(captions)] if modality == "text"
                          else self.paths[(index + offset) % 6]
                          for offset, modality in enumerate(modalities)]
                rows.append([index % 3, *values])
            query_rows[name] = rows
        datasets = _create_query_datasets(query_rows, self.transform)
        self.query_loaders = dict(zip(ORBENCH_QUERY_LOADERS, [
            DataLoader(datasets[name], batch_size=3)
            for name, _, _ in empty._get_modality_combinations()
        ]))

    def test_all_32_configs_match_original_and_encode_unique_inputs_once(self):
        original = Evaluator1(self.gallery_loader, cache_mean_fusion=False, **self.query_loaders)
        cached = Evaluator1(self.gallery_loader, rank_chunk_size=2, **self.query_loaders)
        original_model, cached_model = CountingEncoder(), CountingEncoder()
        with contextlib.redirect_stdout(io.StringIO()):
            original_score = original.eval(original_model, use_weight=False)
            cached_score = cached.eval(cached_model, use_weight=False)
        self.assertEqual(len(cached.last_results), 32)
        self.assertEqual(cached.last_results.keys(), original.last_results.keys())
        np.testing.assert_allclose(cached_score, original_score, atol=1e-5, rtol=1e-6)
        for name in original.last_results:
            np.testing.assert_allclose(
                list(cached.last_results[name].values()), list(original.last_results[name].values()),
                atol=1e-5, rtol=1e-6, err_msg=name,
            )
            loader = self.query_loaders[name.lower().replace('+', '_') + '_query_loader']
            self.assertEqual(cached.last_cache_stats['query_counts'][name], len(loader.dataset))
        for name in original.last_group_results:
            np.testing.assert_allclose(list(cached.last_group_results[name].values()),
                                       list(original.last_group_results[name].values()), atol=1e-5)
        self.assertEqual(dict(cached_model.counts), {"rgb": 12, "nir": 6, "cp": 6, "sk": 6, "text": 3})
        self.assertGreater(sum(original_model.counts.values()), 10 * sum(cached_model.counts.values()))
        # Caches must be rebuilt after model updates, never reused across epochs.
        with contextlib.redirect_stdout(io.StringIO()):
            cached.eval(cached_model, use_weight=False)
        self.assertEqual(cached_model.counts['rgb'], 24)

    def test_chunked_ranking_matches_full_sort_including_ties(self):
        generator = torch.Generator().manual_seed(31)
        gallery = F.normalize(torch.randn(15, 7, generator=generator), dim=1)
        queries = F.normalize(torch.randn(11, 7, generator=generator), dim=1)
        queries[0] = 0  # Tie handling must use the same full argsort operation.
        query_pids = torch.arange(11) % 3
        gallery_pids = torch.arange(15) % 3
        for get_map in (True, False):
            expected = rank(queries @ gallery.T, query_pids, gallery_pids, get_mAP=get_map)[:3]
            for chunk in (1, 4, 256):
                actual = rank_feature_chunks(queries, gallery, query_pids, gallery_pids,
                                             chunk_size=chunk, get_mAP=get_map)
                for actual_metric, expected_metric in zip(actual, expected):
                    torch.testing.assert_close(actual_metric, expected_metric, atol=1e-5,
                                               rtol=1e-6, check_dtype=False)

    def test_fused_features_average_raw_modalities_before_normalizing(self):
        evaluator = Evaluator1(self.gallery_loader, **self.query_loaders)
        combinations = [item for item in evaluator._get_modality_combinations()
                        if item[0] in ("NIR+CP", "TEXT+SK", "NIR+CP+SK+TEXT")]
        model = CountingEncoder()
        _, _, banks, specs, _ = build_mean_feature_cache(
            model, self.gallery_loader, combinations, evaluator.modality_encoders
        )
        for name, modalities, loader in combinations:
            expected_pids, expected_features = evaluator._extract_multi_modality_features(
                model, loader, modalities, use_weight=False
            )
            actual_pids, actual_features = mean_query_features(modalities, banks, specs[name])
            torch.testing.assert_close(actual_pids, expected_pids)
            torch.testing.assert_close(actual_features, expected_features)
            # Modalities deliberately have unequal magnitudes: this guards the
            # tempting but protocol-changing normalization-before-fusion variant.
            normalized_modalities = [F.normalize(banks[modality][index], dim=1)
                                     for modality, index in zip(modalities, specs[name][1])]
            incorrect = F.normalize(torch.stack(normalized_modalities).mean(dim=0), dim=1)
            self.assertFalse(torch.allclose(actual_features, incorrect, atol=1e-4))

    def test_stochastic_preprocessing_does_not_use_metadata_cache(self):
        self.gallery_loader.dataset.transform = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
        evaluator = Evaluator1(self.gallery_loader, **self.query_loaders)
        self.assertFalse(supports_metadata_cache(self.gallery_loader, evaluator._get_modality_combinations()))


if __name__ == "__main__":
    unittest.main()
