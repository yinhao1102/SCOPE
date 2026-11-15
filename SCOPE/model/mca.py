"""Training-only CE-gradient modality contribution alignment.

The reference is equal-weight fusion of detached, raw encoder features.
The target gradient is computed only with respect to temporary fusion weights.
The target does not optimize the encoder or the fusion heads.
"""
import itertools

import torch
import torch.nn.functional as F

from . import objectives


def gallery_context(rgb, pids, bank):
    with torch.no_grad(), torch.autocast(rgb.device.type, enabled=False):
        gallery = F.normalize(torch.cat([
            rgb.detach().float(), bank["features"].detach().float()
        ]), dim=1)
        gallery_ids = torch.cat([pids.detach(), bank["pids"]])
        positive = pids[:, None].eq(gallery_ids[None, :])
        if not positive.any(1).all():
            raise ValueError("Every training query must have a positive candidate")
    return {"gallery": gallery, "positive": positive}


def gradient_contribution(modalities, indices, context, scale):
    """Directional derivative along e_i - uniform on the weight simplex."""
    first = modalities[indices[0]]
    with torch.enable_grad(), torch.autocast(first.device.type, enabled=False):
        reference = torch.full(
            (len(first), len(indices)), 1.0 / len(indices),
            device=first.device, requires_grad=True,
        )
        fused = sum(
            modalities[i].detach().float() * reference[:, j:j + 1]
            for j, i in enumerate(indices)
        )
        scores = F.normalize(fused, dim=1) @ context["gallery"].T
        log_probability = F.log_softmax(scores * scale.detach(), dim=1)
        positive = context["positive"]
        utility = log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)
        gradient = torch.autograd.grad(utility.sum(), reference, create_graph=False)[0]
        contribution = gradient - (gradient * reference).sum(1, keepdim=True)
    return contribution.detach()


def modality_subsets(active_modalities):
    active = tuple(active_modalities)
    return {
        size: list(itertools.combinations(active, size))
        for size in range(2, len(active) + 1)
    }


def fusion_loss(model, features, pids, active_modalities, temperatures=None,
                bank=None, mca_weight=1.5):
    """SDM + lambda MCA; average subsets within each size, then sum sizes."""
    rgb, modalities = features[0], features[1:]
    scale = model.logit_scale.exp().clamp(max=100).detach()
    subsets = modality_subsets(active_modalities)
    retrieval = sum(
        torch.stack([
            objectives.compute_sdm(
                rgb, model.msf_fuse_by_indices(modalities, ids)[0], pids, scale
            )
            for ids in combinations
        ]).mean()
        for combinations in subsets.values()
    )
    if mca_weight == 0:
        return retrieval, {"sdm": float(retrieval.detach()), "mca": 0.0}
    if bank is None or temperatures is None:
        raise ValueError("MCA needs a training reference bank and calibrated temperatures")
    context = gallery_context(rgb, pids, bank)
    targets = {}
    for size, combinations in subsets.items():
        temperature = float(temperatures[str(size)])
        if temperature <= 0:
            raise ValueError("MCA temperatures must be positive")
        for ids in combinations:
            contribution = gradient_contribution(modalities, ids, context, scale)
            targets[ids] = (contribution / temperature).softmax(1).detach()
    # Preserve the archived order: all retrieval forwards, then a second
    # student forward for MCA (including its independent dropout draw).
    alignment = sum(
        torch.stack([
            F.kl_div(
                model.msf_fuse_by_indices(modalities, ids)[1].float().log_softmax(1),
                targets[ids], reduction="none",
            ).sum(1).mean()
            for ids in combinations
        ]).mean()
        for combinations in subsets.values()
    )
    return retrieval + mca_weight * alignment, {
        "sdm": float(retrieval.detach()), "mca": float(alignment.detach()),
    }
