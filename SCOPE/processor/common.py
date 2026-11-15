"""Local, portable runtime helpers; no cluster service is required."""
import hashlib
import json
import csv
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from SCOPE.model.msf import HEADS


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def encoder_fingerprint(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if any(name.startswith(prefix + ".") for prefix in HEADS):
            continue
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), str(value.dtype))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def encode_batch(model, batch, active_modalities):
    device = next(model.parameters()).device
    rgb = model.encode_rgb_cls(batch["rgbs"].to(device, non_blocking=True))
    modalities = [None] * 4
    for i, (name, key) in enumerate([
        ("nir", "nirs"), ("cp", "cps"), ("sk", "sks"), ("text", "caption_ids")
    ]):
        if i in active_modalities:
            modalities[i] = getattr(model, "encode_" + name + "_cls")(
                batch[key].to(device, non_blocking=True)
            )
    return [rgb, *modalities]


def autocast_context(args):
    device = torch.device(args.device)
    enabled = device.type == "cuda" and args.amp_dtype != "none"
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype, enabled=enabled)


def setup_logging(output, name):
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / (name + ".log"))],
    )


def write_metric_tables(output, stem, metrics):
    """All columns are from one selected model, never independent column maxima."""
    write_json(output / (stem + ".json"), metrics)
    columns = ("R1", "R5", "R10", "mAP", "mINP")
    for key in ("per_query", "per_group"):
        with (output / (stem + "_" + key + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("setting", *columns))
            writer.writeheader()
            for setting, values in metrics[key].items():
                writer.writerow({"setting": setting, **{k: values[k] for k in columns}})
