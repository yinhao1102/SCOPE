"""Portable two-stage SCOPE training and evaluation."""
import copy
import json
import logging
import math
import time
from pathlib import Path

import torch
import yaml

from SCOPE.model import SCOPE, objectives
from SCOPE.model.mca import fusion_loss, gallery_context, gradient_contribution, modality_subsets
from SCOPE.model.msf import HEADS, attach_msf, fusion_mode, heads_state, load_heads
from SCOPE.solver import build_lr_scheduler, build_optimizer
from .common import (
    autocast_context, encode_batch, encoder_fingerprint, save_checkpoint, seed_all, write_json,
    write_metric_tables, setup_logging,
)
from SCOPE.utils.options import arguments
from SCOPE.datasets.build import DataBundle


def build_model(args, num_classes):
    return SCOPE.from_pretrained(
        args.pretrain_choice, image_size=tuple(args.img_size),
        rank_magnitude=args.lora_r1, rank_direction=args.lora_r2,
        alpha=args.dual_lora_alpha, temperature=args.temperature,
    ).to(args.device)


def encoder_loss(model, features, pids, active_modalities):
    scale = model.logit_scale.exp().clamp(max=100)
    return sum(
        objectives.compute_sdm(features[0], features[index + 1], pids, scale)
        for index in active_modalities
    )


def optimizer_for_stage(args, model, stage):
    settings = copy.copy(args)
    settings.lr = args.lr if stage == 1 else args.fusion_lr
    settings.num_epoch = args.encoder_epochs if stage == 1 else args.fusion_epochs
    if stage == 2:
        settings.lr_factor = 1.0
    optimizer = build_optimizer(settings, model)
    names = {id(p): name for name, p in model.named_parameters()}
    if stage == 1:
        for group in optimizer.param_groups:
            name = names[id(group["params"][0])]
            if (name.startswith("vision_encoder.") and "lora" in name and "gate" not in name):
                group["lr"] *= args.visual_factor_lr_mult
    scheduler = build_lr_scheduler(settings, optimizer) if stage == 1 else None
    return optimizer, scheduler


def fusion_lr_factor(epoch, epochs):
    if epoch <= 3:
        return 0.1 + 0.9 * (epoch - 1) / 3
    return 0.01 + 0.99 * (1 + math.cos(math.pi * (epoch - 4) / max(1, epochs - 4))) / 2


def calibrate_mca(model, bundle, args, output):
    bank, manifest = bundle.reference_bank(model)
    save_checkpoint(output / "train_reference_bank.pth",
                    {key: value.cpu() for key, value in bank.items()})
    write_json(output / "train_reference_manifest.json", manifest)
    seed_all(args.seed)
    values = {str(size): [] for size in range(2, len(bundle.active) + 1)}
    scale = model.logit_scale.exp().clamp(max=100).detach()
    batches = 0
    for batch in bundle.train_loader(fusion=True):
        with torch.no_grad(), autocast_context(args):
            features = encode_batch(model, batch, bundle.active)
        context = gallery_context(features[0], batch["pids"].to(args.device), bank)
        for size, subsets in modality_subsets(bundle.active).items():
            for indices in subsets:
                values[str(size)].append(gradient_contribution(
                    features[1:], indices, context, scale
                ).cpu())
        batches += 1
        if batches == args.mca_probe_batches:
            break
    if batches != args.mca_probe_batches:
        raise ValueError(
            f"Calibration needs {args.mca_probe_batches} training batches, found {batches}"
        )
    temperatures, rms_values = {}, {}
    for size, parts in values.items():
        contribution = torch.cat(parts).double()
        rms = float((contribution - contribution.mean(1, keepdim=True)).square().mean().sqrt())
        rms_values[size] = rms
        temperatures[size] = rms if rms >= 1e-8 else 1.0
    calibration = {
        "temperatures": temperatures, "centered_contribution_rms": rms_values,
        "probe_batches": batches, "bank_identities": len(bank["pids"]),
        "used_test_data": False, "optimized_during_calibration": False,
        "rule": "Temperature equals training contribution RMS; a degenerate group uses 1",
    }
    write_json(output / "calibration.json", calibration)
    return bank, temperatures


def load_encoder(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" not in state or state.get("format") == "msf_heads":
        raise ValueError("Expected an encoder checkpoint containing a model state dictionary")
    model.load_state_dict(state["model"], strict=True)
    return state


def checkpoint_payload(model, args, stage, metrics, encoder_checkpoint, fingerprint):
    return {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if stage == 1 else heads_state(model),
        "format": "full_encoder" if stage == 1 else "msf_heads",
        "stage": stage, "epoch": metrics["epoch"], "metrics": metrics,
        "config": vars(args).copy(),
        "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
        "encoder_sha256": fingerprint,
    }


def train_stage(model, bundle, args, stage, encoder_checkpoint=None):
    output = Path(args.output_dir) / f"stage{stage}"
    if any(output.glob("metrics_*.jsonl")):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    epochs = args.encoder_epochs if stage == 1 else args.fusion_epochs
    temperatures = bank = fingerprint = None
    caches = {}
    if stage == 2:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        fingerprint = encoder_fingerprint(model)
        # Extract once from the selected encoder before resetting the MSF
        # initialization and training RNG. Never reuse features from another run.
        for split in ("val", "test"):
            cache = bundle.cache(model, split)
            if cache is not None:
                caches[split] = cache
        attach_msf(model, bundle.active, args.seed)
        if args.mca_weight:
            bank, temperatures = calibrate_mca(model, bundle, args, output)
    optimizer, scheduler = optimizer_for_stage(args, model, stage)
    initial_rates = [group["lr"] for group in optimizer.param_groups]
    configuration = dict(vars(args), stage=stage, stage_epochs=epochs,
                         objective="single SDM" if stage == 1 else "multimodal SDM + MCA",
                         selection_split=args.selection_split, strict_encoder_freeze=(stage == 2),
                         encoder_checkpoint=str(encoder_checkpoint) if encoder_checkpoint else None)
    (output / "configs.yaml").write_text(yaml.safe_dump(configuration, sort_keys=True))
    write_json(output / "parameter_counts.json", {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
    })
    seed_all(args.seed)
    loader = bundle.train_loader(fusion=(stage == 2))
    if len(loader) == 0:
        raise ValueError("Training loader is empty; check the data and batch size")
    best_map = best_r1 = -float("inf")
    best = None
    single_baseline = {}
    for epoch in range(1, epochs + 1):
        if stage == 1:
            model.train()
        else:
            fusion_mode(model, True)
            factor = fusion_lr_factor(epoch, epochs)
            for group, rate in zip(optimizer.param_groups, initial_rates):
                group["lr"] = rate * factor
        start = time.time()
        total = 0.0
        for iteration, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args):
                if stage == 1:
                    features = encode_batch(model, batch, bundle.active)
                    loss = encoder_loss(model, features, batch["pids"].to(args.device), bundle.active)
                else:
                    with torch.no_grad():
                        features = encode_batch(model, batch, bundle.active)
                    loss, _ = fusion_loss(
                        model, features, batch["pids"].to(args.device), bundle.active,
                        temperatures, bank, args.mca_weight,
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, iteration {iteration}")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite gradient")
            if stage == 2 and any(
                p.grad is not None and not any(name.startswith(head + ".") for head in HEADS)
                for name, p in model.named_parameters()
            ):
                raise RuntimeError("The fusion stage must not update encoder parameters")
            if epoch == 1 and iteration == 1:
                active = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
                write_json(output / "first_batch.json", {
                    "active_parameter_count": sum(p.numel() for _, p in active),
                    "active_names": [name for name, _ in active],
                })
            optimizer.step()
            total += float(loss.detach())
            if iteration == 1 or iteration % 50 == 0:
                logging.info(f"stage={stage} epoch={epoch}/{epochs} batch={iteration}/{len(loader)} "
                      f"loss={total / iteration:.5f}")
        if scheduler is not None:
            scheduler.step()
        training_seconds = time.time() - start
        for split in ("val", "test"):
            cache = bundle.cache(model, split) if stage == 1 else caches.get(split)
            if cache is None:
                continue
            metrics = bundle.evaluate(model, cache, fused=(stage == 2))
            metrics.update(epoch=epoch, stage=stage, split=split, dataset=args.dataset_name,
                           seed=args.seed, loss=total / len(loader), train_seconds=training_seconds,
                           checkpoint_selection_split=args.selection_split)
            with (output / f"metrics_{split}.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if stage == 2:
                singles = {name: values for name, values in metrics["per_query"].items()
                           if "+" not in name}
                if split not in single_baseline:
                    single_baseline[split] = singles
                elif singles != single_baseline[split]:
                    raise RuntimeError("Single-modal evaluation changed during frozen-encoder training")
            if split != args.selection_split:
                continue
            payload = None
            if metrics["mean_mAP"] > best_map or metrics["mean_R1"] > best_r1:
                payload = checkpoint_payload(
                    model, args, stage, metrics, encoder_checkpoint, fingerprint
                )
            if metrics["mean_mAP"] > best_map:
                best_map, best = metrics["mean_mAP"], metrics
                save_checkpoint(output / "mAP_best.pth", payload)
            if metrics["mean_R1"] > best_r1:
                best_r1 = metrics["mean_R1"]
                save_checkpoint(output / "R1_best.pth", payload)
            logging.info(f"stage={stage} epoch={epoch} {split} Avg-mAP={metrics['mean_mAP']:.4f} "
                  f"best={best_map:.4f}")
        if best is None:
            raise RuntimeError("No metrics were produced for the checkpoint selection split")
        write_json(output / "progress.json", {
            "completed_epochs": epoch, "best_epoch": best["epoch"], "best_mAP": best_map,
        })
    selected = torch.load(output / "mAP_best.pth", map_location="cpu", weights_only=True)
    if stage == 1:
        model.load_state_dict(selected["model"], strict=True)
        cache = bundle.cache(model, args.selection_split)
    else:
        load_heads(model, selected)
        if encoder_fingerprint(model) != fingerprint:
            raise RuntimeError("Encoder parameters or buffers changed in stage two")
        write_json(output / "freeze_verified.json", {
            "encoder_sha256": fingerprint, "single_modal_metrics_unchanged": True,
        })
        cache = caches[args.selection_split]
    reloaded = bundle.evaluate(model, cache, fused=(stage == 2))
    if abs(reloaded["mean_mAP"] - best["mean_mAP"]) > 1e-5:
        raise RuntimeError("Best-checkpoint reload did not reproduce its metrics")
    write_metric_tables(output, "best", best)
    write_json(output / "completed.json", {
        "status": "completed", "epochs": epochs, "best": best, "best_reload_verified": True,
    })
    return best


def prepare(args):
    if args.dry_run:
        print(yaml.safe_dump(vars(args), sort_keys=True))
        return None
    if not Path(args.root_dir).is_dir():
        raise FileNotFoundError("Set --data-root to the dataset parent directory")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; --device cpu supports small smoke checks")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    setup_logging(Path(args.output_dir), "evaluation" if args.evaluation else "train")
    bundle = DataBundle(args)
    if args.check_data:
        print(json.dumps(bundle.check_data(), indent=2))
        return None
    if not Path(args.pretrain_choice).is_file():
        raise FileNotFoundError("Set --ham-checkpoint to the local HAM pretrained file")
    if args.selection_split == "val" and not bundle.has_split("val"):
        raise ValueError("This dataset has no validation queries; select an available split")
    model = build_model(args, bundle.num_classes)
    return model, bundle


def train_main():
    args = arguments()
    prepared = prepare(args)
    if prepared is None:
        return
    model, bundle = prepared
    encoder_checkpoint = getattr(args, "encoder_checkpoint", None)
    if args.stage != "fusion":
        train_stage(model, bundle, args, 1)
        encoder_checkpoint = str(Path(args.output_dir) / "stage1/mAP_best.pth")
    else:
        encoder_checkpoint = encoder_checkpoint or str(Path(args.output_dir) / "stage1/mAP_best.pth")
        load_encoder(model, encoder_checkpoint)
    if args.stage == "fusion" or (args.stage == "all" and not args.no_fusion):
        train_stage(model, bundle, args, 2, encoder_checkpoint)


def evaluate_main():
    args = arguments(evaluation=True)
    prepared = prepare(args)
    if prepared is None:
        return
    model, bundle = prepared
    encoder_checkpoint = getattr(args, "encoder_checkpoint", None)
    encoder_checkpoint = encoder_checkpoint or str(Path(args.output_dir) / "stage1/mAP_best.pth")
    load_encoder(model, encoder_checkpoint)
    fusion_checkpoint = getattr(args, "fusion_checkpoint", None)
    if not fusion_checkpoint and not args.no_fusion:
        candidate = Path(args.output_dir) / "stage2/mAP_best.pth"
        if candidate.is_file():
            fusion_checkpoint = str(candidate)
    if fusion_checkpoint and args.no_fusion:
        raise ValueError("--fusion-checkpoint and --no-fusion cannot be combined")
    if fusion_checkpoint:
        fingerprint = encoder_fingerprint(model)
        attach_msf(model, bundle.active, args.seed)
        state = torch.load(fusion_checkpoint, map_location="cpu", weights_only=True)
        if state.get("encoder_sha256") and state["encoder_sha256"] != fingerprint:
            raise ValueError("Fusion checkpoint belongs to a different encoder")
        load_heads(model, state)
    cache = bundle.cache(model, args.split)
    if cache is None:
        raise ValueError("The requested split is empty")
    result = bundle.evaluate(model, cache, fused=bool(fusion_checkpoint))
    result.update(split=args.split, encoder_checkpoint=str(encoder_checkpoint),
                  fusion_checkpoint=fusion_checkpoint, learned_fusion=bool(fusion_checkpoint))
    write_metric_tables(Path(args.output_dir), "evaluation", result)
    if args.export_rankings:
        bundle.export_rankings(model, cache, Path(args.output_dir) / "rankings.jsonl",
                               fused=bool(fusion_checkpoint), top_k=args.top_k)
    print(json.dumps(result["per_group"], indent=2))
    print("Avg-mAP:", result["mean_mAP"])
