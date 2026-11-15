"""Dataset adapters preserving the native ORBench and Tri query protocols."""
import hashlib
import copy
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from SCOPE.model.msf import MODALITIES, fusion_mode
from SCOPE.datasets.orbench import build_reid5o_dataloader
from SCOPE.datasets.orbench.build import build_transforms as orbench_transforms
from SCOPE.datasets.orbench.build import collate as orbench_collate
from SCOPE.datasets.tri import TriPersonDataset, TriTrainDataset, build_transforms, collate
from SCOPE.utils.cached_orbench_eval import build_mean_feature_cache, rank_feature_chunks
from SCOPE.utils.evaluation import build_orbench_evaluator
from SCOPE.utils.iotools import read_image
from SCOPE.utils.evaluation_tri import encode_tri, evaluate_tri, tri_eval_loaders

METRICS = ("R1", "R5", "R10", "mAP", "mINP")


class ReferenceImages(Dataset):
    def __init__(self, pids, paths, transform):
        self.pids, self.paths, self.transform = pids, paths, transform

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        return self.pids[index], self.transform(read_image(self.paths[index]))


class DataBundle:
    def __init__(self, args):
        self.args = args
        self.is_orbench = args.dataset_name == "ORBench"
        self.active = (0, 1, 2, 3) if self.is_orbench else (2, 3)
        if self.is_orbench:
            settings = copy.copy(args)
            settings.training = True
            loaders = build_reid5o_dataloader(settings)
            self.train_dataset = loaders[0].dataset
            self.train_rows = self.train_dataset.dataset
            self.num_classes = loaders[-1]
            self.evaluator = build_orbench_evaluator(loaders, training=True)
        else:
            self.dataset = TriPersonDataset(args.root_dir, args.dataset_name)
            self.train_rows = self.dataset.train
            self.num_classes = self.dataset.num_train_ids
            self.train_dataset = TriTrainDataset(
                self.train_rows, build_transforms(args.img_size, args.img_aug, True),
                args.text_length,
            )

    def has_split(self, split):
        return split == "test" if self.is_orbench else bool(getattr(self.dataset, split)["image_paths"])

    def check_data(self):
        paths = {str(row[i]) for row in self.train_rows
                 for i in (range(2, 6) if self.is_orbench else (2, 3))}
        if self.is_orbench:
            paths.update(self.evaluator.gallery_loader.dataset.img_paths)
            for _, modalities, loader in self.evaluator._get_modality_combinations():
                paths.update(row[i + 1] for row in loader.dataset.queries
                             for i, modality in enumerate(modalities) if modality != "text")
        else:
            for split in ("val", "test"):
                data = getattr(self.dataset, split)
                paths.update(data["image_paths"])
                paths.update(data["sketch_paths"])
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} dataset files are missing; first: {missing[0]}")
        # Check collating even when the dataset is smaller than the configured batch.
        collator = orbench_collate if self.is_orbench else collate
        batch = collator([self.train_dataset[0]])
        return {"status": "passed", "dataset": self.args.dataset_name,
                "train_samples": len(self.train_dataset), "identities": self.num_classes,
                "checked_files": len(paths),
                "batch_shapes": {k: list(v.shape) for k, v in batch.items()}}

    def train_loader(self, fusion=False):
        return DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
            drop_last=not self.is_orbench,
            num_workers=self.args.fusion_workers if fusion else self.args.num_workers,
            collate_fn=orbench_collate if self.is_orbench else collate,
            pin_memory=torch.device(self.args.device).type == "cuda",
            generator=torch.Generator().manual_seed(self.args.seed),
        )

    @torch.no_grad()
    def cache(self, model, split="test"):
        model.eval()
        if not self.is_orbench:
            data = getattr(self.dataset, split)
            if not data["image_paths"]:
                return None
            cache = encode_tri(model, tri_eval_loaders(data, self.args))
            cache["gallery_paths"] = data["image_paths"]
            return cache
        if split != "test":
            return None
        gids, gallery, banks, specs, stats = build_mean_feature_cache(
            model, self.evaluator.gallery_loader,
            self.evaluator._get_modality_combinations(), self.evaluator.modality_encoders,
        )
        return {"gids": gids, "gallery": gallery, "banks": banks, "specs": specs, "stats": stats,
                "gallery_paths": self.evaluator.gallery_loader.dataset.img_paths}

    @torch.no_grad()
    def evaluate(self, model, cache, fused=False):
        if not self.is_orbench:
            return evaluate_tri(model, cache, fused)
        fusion_mode(model, False)
        per_query = {}
        for name, (pids, indices) in cache["specs"].items():
            modalities = name.lower().split("+")
            features = [cache["banks"][modality][index]
                        for modality, index in zip(modalities, indices)]
            if len(features) == 1:
                query = features[0]
            elif fused:
                inputs = [None] * 4
                active = []
                for modality, feature in zip(modalities, features):
                    index = MODALITIES.index(modality)
                    inputs[index] = feature
                    active.append(index)
                query = model.msf_fuse_by_indices(inputs, active)[0]
            else:
                query = torch.stack(features).mean(0)
            cmc, ap, inp = rank_feature_chunks(
                F.normalize(query, dim=1), cache["gallery"], pids, cache["gids"], chunk_size=256
            )
            per_query[name] = dict(zip(METRICS, [
                float(cmc[0]), float(cmc[min(4, len(cmc) - 1)]),
                float(cmc[min(9, len(cmc) - 1)]), float(ap), float(inp),
            ]))
        if len(per_query) != 32:
            raise ValueError("Expected all 32 native ORBench query settings")
        groups = {}
        for size, name in enumerate(("ONE_AVER", "TWO_AVER", "THREE_AVER", "FOUR_AVER"), 1):
            entries = [value for key, value in per_query.items() if len(key.split("+")) == size]
            groups[name] = {metric: sum(v[metric] for v in entries) / len(entries)
                            for metric in METRICS}
        return {
            "per_query": per_query, "per_group": groups,
            "mean_mAP": sum(v["mAP"] for v in groups.values()) / 4,
            "mean_R1": sum(v["R1"] for v in groups.values()) / 4,
            "selection": "Equal-weight mean of the four native search-group mAP values",
        }

    @torch.no_grad()
    def reference_bank(self, model):
        choices = {}
        for row in self.train_rows:
            pid, path = int(row[0]), str(row[2])
            choices[pid] = min(path, choices.get(pid, path))
        identities = sorted(
            choices, key=lambda pid: hashlib.sha256(("mca-reference-v1|" + str(pid)).encode()).digest()
        )[:self.args.mca_bank_identities]
        if len(identities) < 2:
            raise ValueError("MCA needs at least two training identities")
        paths = [choices[pid] for pid in identities]
        transform = (orbench_transforms(self.args.img_size, is_train=False)
                     if self.is_orbench else build_transforms(self.args.img_size, training=False))
        loader = DataLoader(
            ReferenceImages(identities, paths, transform), batch_size=128, shuffle=False,
            num_workers=self.args.fusion_workers,
            generator=torch.Generator().manual_seed(self.args.seed),
        )
        model.eval()
        device = next(model.parameters()).device
        features, pids = [], []
        for ids, images in loader:
            with torch.autocast(device.type, enabled=False):
                features.append(model.encode_rgb_cls(images.to(device)).float())
            pids.append(ids.to(device))
        return {"features": torch.cat(features), "pids": torch.cat(pids)}, {
            "source": "training RGB only", "used_test_data": False,
            "identity_count": len(identities), "training_ids": identities,
            "paths": [str(Path(path).relative_to(self.args.root_dir)) for path in paths],
        }

    @torch.no_grad()
    def export_rankings(self, model, cache, output, fused=False, top_k=10):
        """Export ranked gallery images using the same source-image exclusion."""
        model.eval()
        gallery = cache["gallery"] if self.is_orbench else F.normalize(cache["gallery"], dim=1)
        gallery_ids = (cache["gids"].cpu().numpy() if self.is_orbench else cache["gallery_pids"])

        def cases():
            if self.is_orbench:
                for name, (pids, indices) in cache["specs"].items():
                    names = name.lower().split("+")
                    parts = [cache["banks"][n][idx] for n, idx in zip(names, indices)]
                    if len(parts) == 1:
                        query = parts[0]
                    elif fused:
                        inputs, active = [None] * 4, []
                        for n, feature in zip(names, parts):
                            i = MODALITIES.index(n)
                            inputs[i] = feature
                            active.append(i)
                        query = model.msf_fuse_by_indices(inputs, active)[0]
                    else:
                        query = torch.stack(parts).mean(0)
                    yield name, query, pids.cpu().numpy(), None, False
            else:
                if fused:
                    pieces = [model.msf_fuse_by_indices(
                        [None, None, cache["paired_sketch"][i:i + 512], cache["text"][i:i + 512]], (2, 3)
                    )[0] for i in range(0, len(cache["text"]), 512)]
                    combined = torch.cat(pieces)
                else:
                    combined = (cache["paired_sketch"] + cache["text"]) / 2
                for name, query, key, exclude in (
                    ("TEXT_original", cache["text"], "text", False),
                    ("TEXT", cache["text"], "text", True),
                    ("SK", cache["sketch"], "sketch", True),
                    ("TEXT+SK", combined, "text", True),
                ):
                    yield name, query, cache[key + "_pids"], cache[key + "_ids"], exclude

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as stream:
            for name, queries, qids, source_ids, exclude in cases():
                for start in range(0, len(queries), 256):
                    scores = F.normalize(queries[start:start + 256], dim=1) @ gallery.T
                    values = scores.cpu().numpy()
                    orders = (scores.argsort(dim=1, descending=True).cpu().numpy()
                              if self.is_orbench else np.argsort(-values, axis=1))
                    for i, order in enumerate(orders):
                        row = start + i
                        if exclude:
                            order = order[~((gallery_ids[order] == qids[row]) &
                                            (cache["gallery_ids"][order] == source_ids[row]))]
                        matches = [{"path": str(Path(cache["gallery_paths"][j]).relative_to(self.args.root_dir)),
                                    "pid": int(gallery_ids[j]), "score": float(values[i, j])}
                                   for j in order[:top_k]]
                        stream.write(json.dumps({"setting": name, "query_index": row,
                                                 "query_pid": int(qids[row]), "gallery": matches}) + "\n")
