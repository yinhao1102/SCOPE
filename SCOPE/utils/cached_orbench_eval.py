"""Reuse deterministic ORBench inputs within one mean-fusion evaluation.

Caches are rebuilt on every evaluation so updated encoder weights are respected.
Raw embeddings are cached: normalizing each modality before averaging would
change the original retrieval protocol.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class _UniqueInputs(Dataset):
    def __init__(self, inputs, text=False):
        self.inputs = inputs
        self.text = text

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.text:
            return self.inputs[index]
        from SCOPE.utils.iotools import read_image
        path, transform = self.inputs[index]
        image = read_image(path)
        return image if transform is None else transform(image)


def _deterministic(transform):
    if type(transform).__name__.startswith("Random"):
        return False
    return all(_deterministic(child) for child in getattr(transform, "transforms", ()))


def supports_metadata_cache(gallery_loader, combinations):
    gallery = gallery_loader.dataset
    if not all(hasattr(gallery, name) for name in ("image_pids", "img_paths", "transform")):
        return False
    if not _deterministic(gallery.transform):
        return False
    for _, modalities, loader in combinations:
        if loader is None:
            continue
        dataset = loader.dataset
        if not hasattr(dataset, "queries"):
            return False
        if any(modality != "text" for modality in modalities):
            if not hasattr(dataset, "transform") or not _deterministic(dataset.transform):
                return False
        if "text" in modalities and not all(
            hasattr(dataset, name) for name in ("tokenizer", "text_length", "truncate")
        ):
            return False
    return True


@torch.no_grad()
def build_mean_feature_cache(model, gallery_loader, combinations, modality_encoders):
    from SCOPE.datasets.orbench.bases import tokenize

    device = next(model.parameters()).device
    inputs = {modality: [] for modality in ("rgb", "nir", "cp", "sk", "text")}
    lookups = {modality: {} for modality in inputs}
    token_keys = {}
    source_loaders = {"rgb": gallery_loader}

    def register(modality, value, dataset):
        if modality == "text":
            description = (value, dataset.text_length, dataset.truncate)
            if description not in token_keys:
                tokens = tokenize(value, dataset.tokenizer, dataset.text_length, dataset.truncate)
                token_keys[description] = tuple(tokens.tolist())
            key = token_keys[description]
            item = None
        else:
            # Different preprocessing objects get separate entries, even if the
            # path is shared. Native ORBench uses a single deterministic transform.
            key = (str(value), id(dataset.transform))
            item = (value, dataset.transform)
        if key not in lookups[modality]:
            lookups[modality][key] = len(inputs[modality])
            if modality == "text":
                item = torch.tensor(key, dtype=torch.long)
            inputs[modality].append(item)
        return lookups[modality][key]

    gallery = gallery_loader.dataset
    gallery_indices = [register("rgb", path, gallery) for path in gallery.img_paths]
    query_specs = {}
    for name, modalities, loader in combinations:
        if loader is None:
            continue
        dataset = loader.dataset
        indices = [[] for _ in modalities]
        pids = []
        for row in dataset.queries:
            pids.append(row[0])
            for column, modality in enumerate(modalities):
                source_loaders.setdefault(modality, loader)
                indices[column].append(register(modality, row[column + 1], dataset))
        query_specs[name] = (torch.tensor(pids, device=device), [
            torch.tensor(index, dtype=torch.long, device=device) for index in indices
        ])

    feature_banks = {}
    for modality, values in inputs.items():
        if not values:
            continue
        source = source_loaders[modality]
        loader = DataLoader(
            _UniqueInputs(values, text=modality == "text"),
            batch_size=source.batch_size or 1,
            shuffle=False,
            num_workers=source.num_workers,
            pin_memory=source.pin_memory,
        )
        encoder_name = "encode_rgb_cls" if modality == "rgb" else modality_encoders[modality]
        encoder = getattr(model, encoder_name)
        feature_banks[modality] = torch.cat([
            # CLS can be a view into all token outputs. Keep only the compact
            # embeddings, rather than retaining every token until cat finishes.
            encoder(batch.to(device)).contiguous() for batch in loader
        ])
    gallery_features = feature_banks["rgb"][gallery_indices]
    gallery_pids = torch.tensor(gallery.image_pids, device=device)
    stats = {"encoded_samples": {key: len(value) for key, value in inputs.items()},
             "query_counts": {name: len(spec[0]) for name, spec in query_specs.items()}}
    return gallery_pids, F.normalize(gallery_features, dim=1), feature_banks, query_specs, stats


def mean_query_features(modalities, feature_banks, spec):
    pids, indices = spec
    features = [feature_banks[modality][index] for modality, index in zip(modalities, indices)]
    raw_features = features[0] if len(features) == 1 else torch.stack(features).mean(dim=0)
    return pids, F.normalize(raw_features, dim=1)


@torch.no_grad()
def rank_feature_chunks(query_features, gallery_features, query_pids, gallery_pids,
                        chunk_size=256, max_rank=10, get_mAP=True):
    """Original full-gallery PID metrics, bounded by query chunk size.

All gallery entries are sorted for every query; no top-k approximation, camera
filtering, positive removal, or query subsampling is performed.
"""
    if chunk_size < 1 or query_features.shape[0] < 1:
        raise ValueError("Ranking needs nonempty queries and a positive chunk size")
    device = query_features.device
    query_pids, gallery_pids = query_pids.to(device), gallery_pids.to(device)
    rank_count = min(max_rank, gallery_features.shape[0])
    cmc_total = torch.zeros(rank_count, dtype=torch.float64, device=device)
    ap_total = torch.zeros((), dtype=torch.float64, device=device)
    inp_total = torch.zeros_like(ap_total)
    positions = torch.arange(1, gallery_features.shape[0] + 1, device=device)
    for start in range(0, len(query_features), chunk_size):
        similarities = query_features[start:start + chunk_size] @ gallery_features.t()
        if get_mAP:
            order = similarities.argsort(dim=1, descending=True)
        else:
            order = similarities.topk(rank_count, dim=1, largest=True, sorted=True).indices
        matches = gallery_pids[order].eq(query_pids[start:start + chunk_size, None])
        cumulative = matches.cumsum(dim=1)
        cmc_total += cumulative[:, :rank_count].clamp(max=1).sum(dim=0)
        if get_mAP:
            positives = matches.sum(dim=1)
            if torch.any(positives == 0):
                raise RuntimeError("ORBench query identity has no gallery positive")
            precisions = cumulative / positions
            ap_total += ((precisions * matches).sum(dim=1) / positives).double().sum()
            last_positive = torch.where(matches, positions, 0).amax(dim=1)
            inp_total += (positives / last_positive).double().sum()
    scale = 100.0 / len(query_features)
    return (cmc_total * scale).float(), (ap_total * scale).float(), (inp_total * scale).float()
