"""Shared, portable training and evaluation configuration."""
import argparse
from pathlib import Path

import yaml

DATASETS = ("ORBench", "Tri-CUHK-PEDES", "Tri-RSTPReid", "Tri-ICFG-PEDES")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/ham_orbench.yaml"


def load_yaml(path, visited=None):
    path = Path(path).expanduser().resolve()
    visited = set() if visited is None else visited
    if path in visited:
        raise ValueError("Cyclic configuration inheritance")
    visited.add(path)
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("A configuration must be a YAML mapping")
    parent = values.pop("_base_", None)
    base = load_yaml(path.parent / parent, visited) if parent else {}
    base.update(values)
    return base


def arguments(evaluation=False, argv=None):
    parser = argparse.ArgumentParser(description="HAM-based SCOPE training and retrieval evaluation")
    parser.add_argument("--config", "--config-file", "--config_file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", dest="root_dir")
    parser.add_argument("--ham-checkpoint", dest="pretrain_choice")
    parser.add_argument("--output", dest="output_dir")
    parser.add_argument("--dataset", dest="dataset_name", choices=DATASETS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--test-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rank", type=int, help="Set both GD-LoRA branch ranks")
    parser.add_argument("--mca-weight", type=float)
    parser.add_argument("--mca-probe-batches", type=int)
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--fusion-checkpoint")
    parser.add_argument("--stage", choices=("all", "encoder", "fusion"), default="all")
    parser.add_argument("--switch-epoch", "--encoder-epochs", dest="encoder_epochs", type=int,
                        help="Complete N encoder epochs, then start stage 2 at global epoch N+1")
    parser.add_argument("--fusion-epochs", type=int, help="Number of epochs after the stage switch")
    parser.add_argument("--selection-split", choices=("test", "val"))
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--export-rankings", action="store_true", help="Write top-k retrievals as JSONL")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--check-data", action="store_true", help="Check files and a collated batch, without loading HAM")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration")
    cli = parser.parse_args(argv)
    # Saved run configs and dataset presets share the same defaults.
    cfg = load_yaml(DEFAULT_CONFIG.parent / "defaults.yaml")
    cfg.update(load_yaml(cli.config))
    for key in ("root_dir", "pretrain_choice", "output_dir", "dataset_name", "device", "seed",
                "batch_size", "test_batch_size", "encoder_epochs", "fusion_epochs", "mca_weight",
                "mca_probe_batches", "encoder_checkpoint", "fusion_checkpoint", "selection_split"):
        value = getattr(cli, key)
        if value is not None:
            cfg[key] = value
    if cli.workers is not None:
        cfg["num_workers"] = cfg["fusion_workers"] = cli.workers
    if cli.rank is not None:
        cfg["lora_r1"] = cfg["lora_r2"] = cli.rank
    for key in ("root_dir", "pretrain_choice", "output_dir", "encoder_checkpoint", "fusion_checkpoint"):
        if cfg.get(key):
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    cfg.update(stage=cli.stage, split=cli.split, no_fusion=cli.no_fusion, evaluation=evaluation,
               dry_run=cli.dry_run, check_data=cli.check_data, export_rankings=cli.export_rankings,
               top_k=cli.top_k, training=not evaluation, distributed=False)
    if cfg["dataset_name"] not in DATASETS:
        parser.error("Unsupported dataset")
    if cfg["amp_dtype"] not in ("bf16", "none"):
        parser.error("amp_dtype must be bf16 or none")
    if cfg["dataset_name"] == "ORBench" and (cfg["selection_split"] == "val" or cli.split == "val"):
        parser.error("The native ORBench annotations provide train/test splits only")
    positive = ("lora_r1", "lora_r2", "batch_size", "test_batch_size", "encoder_epochs",
                "fusion_epochs", "mca_probe_batches", "mca_bank_identities", "top_k", "threads")
    if any(cfg[key] < 1 for key in positive):
        parser.error("Ranks, batch sizes, epoch budgets, and count settings must be positive")
    if cfg["mca_weight"] < 0 or cfg["num_workers"] < 0 or cfg["fusion_workers"] < 0:
        parser.error("MCA weight and worker counts must be nonnegative")
    return argparse.Namespace(**cfg)
