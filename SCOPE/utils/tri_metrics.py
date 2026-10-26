import logging

import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def evaluate_retrieval(similarity, query_pids, gallery_pids, query_ids=None, gallery_ids=None):
    order = torch.argsort(similarity, dim=1, descending=True).cpu().numpy()
    query_pids = query_pids.cpu().numpy()
    gallery_pids = gallery_pids.cpu().numpy()
    query_ids = None if query_ids is None else query_ids.cpu().numpy()
    gallery_ids = None if gallery_ids is None else gallery_ids.cpu().numpy()

    cmcs, average_precisions = [], []
    for index, ranked_indices in enumerate(order):
        if query_ids is not None and gallery_ids is not None:
            ranked_indices = ranked_indices[gallery_ids[ranked_indices] != query_ids[index]]
        matches = gallery_pids[ranked_indices] == query_pids[index]
        if not np.any(matches):
            continue
        cmc = np.minimum(1, np.cumsum(matches))
        precisions = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        cmcs.append(cmc[:10])
        average_precisions.append((precisions * matches).sum() / matches.sum())

    if not cmcs:
        raise RuntimeError("No valid query has a matching gallery identity.")
    cmcs = [np.pad(cmc, (0, 10 - len(cmc)), mode="edge") for cmc in cmcs]
    return np.mean(cmcs, axis=0) * 100, float(np.mean(average_precisions) * 100)


class TriEvaluator:
    def __init__(self, gallery_loader, text_sketch_loader, sketch_loader):
        self.gallery_loader = gallery_loader
        self.text_sketch_loader = text_sketch_loader
        self.sketch_loader = sketch_loader
        self.logger = logging.getLogger("SCOPE.tri_eval")

    @torch.no_grad()
    def _compute_embeddings(self, model, use_weight):
        device = next(model.parameters()).device
        text_features, paired_sketch_features, fused_features = [], [], []
        text_pids, text_image_ids = [], []

        for sketches, image_ids, pids, captions in self.text_sketch_loader:
            sketches, captions = sketches.to(device), captions.to(device)
            text_feature = model.encode_text_cls(captions)
            sketch_feature = model.encode_sk_cls(sketches)
            if use_weight:
                modal_features = [None, None, sketch_feature, text_feature]
                fused_feature, _, _ = model.msf_fuse_by_indices(modal_features, (2, 3))
            else:
                fused_feature = (sketch_feature + text_feature) / 2
            text_features.append(text_feature)
            paired_sketch_features.append(sketch_feature)
            fused_features.append(fused_feature)
            text_pids.append(pids)
            text_image_ids.append(image_ids)

        sketch_features, sketch_pids, sketch_image_ids = [], [], []
        for pids, sketches, image_ids in self.sketch_loader:
            sketch_features.append(model.encode_sk_cls(sketches.to(device)))
            sketch_pids.append(pids)
            sketch_image_ids.append(image_ids)

        gallery_features, gallery_pids, gallery_image_ids = [], [], []
        for pids, images, image_ids in self.gallery_loader:
            gallery_features.append(model.encode_rgb_cls(images.to(device)))
            gallery_pids.append(pids)
            gallery_image_ids.append(image_ids)

        return {
            "text": torch.cat(text_features),
            "paired_sketch": torch.cat(paired_sketch_features),
            "fused": torch.cat(fused_features),
            "text_pids": torch.cat(text_pids),
            "text_ids": torch.cat(text_image_ids),
            "sketch": torch.cat(sketch_features),
            "sketch_pids": torch.cat(sketch_pids),
            "sketch_ids": torch.cat(sketch_image_ids),
            "gallery": torch.cat(gallery_features),
            "gallery_pids": torch.cat(gallery_pids),
            "gallery_ids": torch.cat(gallery_image_ids),
        }

    def eval(self, model, use_weight=False):
        features = self._compute_embeddings(model.eval(), use_weight)
        gallery = F.normalize(features["gallery"], dim=1)
        results = {}
        queries = {
            "Text -> RGB": (features["text"], features["text_pids"], features["text_ids"]),
            "Sketch -> RGB": (features["sketch"], features["sketch_pids"], features["sketch_ids"]),
            "Text+Sketch -> RGB": (features["fused"], features["text_pids"], features["text_ids"]),
        }
        for name, (query, pids, image_ids) in queries.items():
            similarity = F.normalize(query, dim=1) @ gallery.t()
            results[name] = evaluate_retrieval(
                similarity, pids, features["gallery_pids"], image_ids, features["gallery_ids"]
            )

        table = PrettyTable(["Query", "R1", "R5", "R10", "mAP"])
        for name, (cmc, mean_ap) in results.items():
            table.add_row([name, cmc[0], cmc[4], cmc[9], mean_ap])
        table.float_format = ".4"
        self.logger.info("\n%s", table)

        selected = [results["Text -> RGB"], results["Sketch -> RGB"], results["Text+Sketch -> RGB"]]
        return float(np.mean([item[0][0] for item in selected])), float(np.mean([item[1] for item in selected]))
