import argparse
import csv
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from types import MethodType

import torch
import torch.nn.functional as F
import yaml
from torch.utils.flop_counter import FlopCounterMode


ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "scope": {
        "repo": ROOT,
        "config": Path(
            "/mnt/18T/yinhao/results/logs/ORBench/"
            "20260414_171241_baseline/configs.yaml"
        ),
    },
    "unireid": {
        "repo": ROOT,
        "config": Path(
            "/mnt/18T/yinhao/results/logs/ORBench/"
            "20260317_105038_baseline/configs.yaml"
        ),
    },
    "reid5o": {
        "repo": Path("/home/yinhao/ReID5o_ORBench/ReID5o"),
        "config": Path(
            "/home/yinhao/ReID5o_ORBench/ReID5o/logs/"
            "reid5o_ckpt/configs.yaml"
        ),
    },
    "irra": {
        "repo": Path("/home/yinhao/IRRA_profile_clean"),
        "config": Path(
            "/home/yinhao/IRRA/logs/CUHK-PEDES/"
            "20250729_160801_irra/configs.yaml"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile one real training iteration (forward + backward + optimizer)."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("scope", "unireid", "reid5o", "irra"),
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2"),
        default="stage1",
        help="Only affects SCOPE.",
    )
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override an outdated root_dir stored in an old config.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "real_training_flops.csv",
    )
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.load(handle, Loader=yaml.UnsafeLoader)
    return SimpleNamespace(**values)


def prepare_imports(repo):
    os.chdir(repo)
    sys.path.insert(0, str(repo))


def bind_irra_orbench_forward(model, objectives):
    """Restore the original PRCV baseline: IRRA trained on all ORBench modalities."""

    def forward(self, batch):
        outputs = {}
        caption_ids = batch["caption_ids"]
        image_feats, text_feats = self.base_model(batch["rgbs"], caption_ids)
        nir_feats = self.base_model.encode_image(batch["nirs"])
        cp_feats = self.base_model.encode_image(batch["cps"])
        sk_feats = self.base_model.encode_image(batch["sks"])

        rgb_global = image_feats[:, 0, :].float()
        query_globals = [
            text_feats[
                torch.arange(text_feats.shape[0], device=text_feats.device),
                caption_ids.argmax(dim=-1),
            ].float(),
            nir_feats[:, 0, :].float(),
            cp_feats[:, 0, :].float(),
            sk_feats[:, 0, :].float(),
        ]
        logit_scale = self.logit_scale
        outputs["temperature"] = 1 / logit_scale

        if "sdm" in self.current_task:
            outputs["sdm_loss"] = torch.stack(
                [
                    objectives.compute_sdm(
                        rgb_global, query, batch["pids"], logit_scale
                    )
                    for query in query_globals
                ]
            ).mean()

        if "id" in self.current_task:
            logits = [
                self.classifier(feature)
                for feature in [rgb_global, *query_globals]
            ]
            outputs["id_loss"] = torch.stack(
                [F.cross_entropy(logit, batch["pids"]) for logit in logits]
            ).mean() * self.args.id_loss_weight

        if "mlm" in self.current_task:
            mlm_feats = self.base_model.encode_text(batch["mlm_ids"])
            cross_feats = self.cross_former(mlm_feats, image_feats, image_feats)
            scores = self.mlm_head(cross_feats).float().reshape(
                -1, self.args.vocab_size
            )
            labels = batch["mlm_labels"].reshape(-1)
            outputs["mlm_loss"] = (
                objectives.compute_mlm(scores, labels)
                * self.args.mlm_loss_weight
            )

        return outputs

    model.forward = MethodType(forward, model)


def build_training_objects(method, args):
    if method in {"scope", "unireid"}:
        from reid5o_datasets import build_reid5o_dataloader
        from model import build_lora_model, build_model
        from solver import build_optimizer

        dataloaders = build_reid5o_dataloader(args)
        train_loader = dataloaders[0]
        num_classes = dataloaders[-1]
        if method == "scope":
            model = build_lora_model(args, num_classes)
        else:
            model = build_model(args, num_classes)
    elif method == "reid5o":
        from datasets import build_dataloader
        from model import build_model
        from solver import build_optimizer

        dataloaders = build_dataloader(args)
        train_loader = dataloaders[0]
        num_classes = dataloaders[-1]
        model = build_model(args, num_classes)
    else:
        from model import objectives
        from model import build_model
        from solver import build_optimizer

        # IRRA itself only ships text-ReID dataloaders. PRCV's initial
        # ORBench baseline reused IRRA with RGB/NIR/CP/SK/TEXT batches.
        data_args = load_config(DEFAULTS["unireid"]["config"])
        data_args.batch_size = args.batch_size
        data_args.num_workers = 0
        data_args.distributed = False
        data_args.training = True
        data_args.dataset_name = "ORBench"
        data_args.root_dir = args.root_dir
        data_args.pretrain_choice = "ViT-B/16"
        data_args.img_size = (384, 128)
        data_args.stride_size = 16
        data_args.text_length = 77

        sys.path.insert(0, str(ROOT))
        from reid5o_datasets import build_reid5o_dataloader

        dataloaders = build_reid5o_dataloader(data_args)
        train_loader = dataloaders[0]
        num_classes = dataloaders[-1]

        args.loss_names = "sdm+id+mlm"
        model = build_model(args, num_classes)
        bind_irra_orbench_forward(model, objectives)

    return train_loader, model, build_optimizer


def configure_scope_stage(model, args, stage):
    if stage == "stage1":
        args.use_weight = False
        args.mm_itc_loss_weight = 0.0
        args.single_itc_loss_weight = 1.0
        for name, parameter in model.named_parameters():
            if "adapter" in name:
                parameter.requires_grad = False
        return

    args.use_weight = True
    args.mm_itc_loss_weight = 1.0
    args.single_itc_loss_weight = 0.0
    for name, parameter in model.named_parameters():
        if "lora" in name or "tokenizer" in name or "ln" in name:
            parameter.requires_grad = False
        if "adapter" in name:
            parameter.requires_grad = True


def move_batch_to_cpu(batch):
    if not isinstance(batch, dict):
        raise TypeError(f"Expected a dict batch, got {type(batch)!r}")
    return {
        ("rgbs" if key == "images" else key): value.cpu()
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def add_irra_mlm_fields(batch):
    """Apply IRRA's original 15% BERT-style token masking."""
    mlm_ids = batch["caption_ids"].clone()
    mlm_labels = torch.zeros_like(mlm_ids)
    token_range = range(1, 49405)

    for row in range(mlm_ids.shape[0]):
        masked_any = False
        for column in range(mlm_ids.shape[1]):
            token = int(mlm_ids[row, column])
            if not 0 < token < 49405 or random.random() >= 0.15:
                continue

            mlm_labels[row, column] = token
            masked_any = True
            replacement_probability = random.random()
            if replacement_probability < 0.8:
                mlm_ids[row, column] = 49405
            elif replacement_probability < 0.9:
                mlm_ids[row, column] = random.choice(token_range)

        if not masked_any:
            mlm_labels[row, 1] = mlm_ids[row, 1]
            mlm_ids[row, 1] = 49405

    batch["mlm_ids"] = mlm_ids
    batch["mlm_labels"] = mlm_labels


def collect_loss(outputs):
    if torch.is_tensor(outputs):
        return outputs
    losses = [
        value
        for name, value in outputs.items()
        if "loss" in name and torch.is_tensor(value)
    ]
    if not losses:
        raise RuntimeError(f"No tensor loss found in outputs: {outputs.keys()}")
    return sum(losses)


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    cli = parse_args()
    defaults = DEFAULTS[cli.method]
    repo = (cli.repo or defaults["repo"]).resolve()
    config = (cli.config or defaults["config"]).resolve()
    prepare_imports(repo)

    args = load_config(config)
    args.batch_size = cli.batch_size
    args.num_workers = 0
    args.distributed = False
    args.training = True
    args.pretrain_choice = "ViT-B/16"
    args.img_size = (384, 128)
    args.stride_size = 16
    args.text_length = 77
    if cli.data_root is not None:
        args.root_dir = str(cli.data_root.resolve())
    elif cli.method == "irra":
        args.root_dir = "/home/yinhao/data"

    train_loader, model, optimizer_builder = build_training_objects(cli.method, args)
    if cli.method == "scope":
        configure_scope_stage(model, args, cli.stage)

    model = model.float().cpu().train()
    optimizer = optimizer_builder(args, model)
    batch = move_batch_to_cpu(next(iter(train_loader)))
    if cli.method == "irra" and "mlm_ids" not in batch:
        add_irra_mlm_fields(batch)
    actual_batch_size = next(
        value.shape[0]
        for value in batch.values()
        if torch.is_tensor(value) and value.ndim > 0
    )

    optimizer.zero_grad(set_to_none=True)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        outputs = model(batch)
        loss = collect_loss(outputs)
        loss.backward()
        optimizer.step()

    total_flops = int(flop_counter.get_total_flops())
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    label = (
        f"{cli.method}-{cli.stage}"
        if cli.method == "scope"
        else cli.method
    )
    row = {
        "method": label,
        "config": str(config),
        "backbone": "ViT-B/16",
        "image_size": "384x128",
        "text_length": 77,
        "batch_size": actual_batch_size,
        "total_flops": total_flops,
        "tflops_per_batch": f"{total_flops / 1e12:.6f}",
        "gflops_per_sample": f"{total_flops / actual_batch_size / 1e9:.6f}",
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    append_result(cli.output, row)

    print("=" * 72)
    for key, value in row.items():
        print(f"{key}: {value}")
    print(f"saved_to: {cli.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
