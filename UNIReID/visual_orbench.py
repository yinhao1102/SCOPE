import argparse
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import fill

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATES = [
    Path.cwd(),
    SCRIPT_DIR,
    SCRIPT_DIR / "UNIReID",
    Path("/home/yinhao/UNIReID"),
]

PROJECT_ROOT = None
for candidate in PROJECT_ROOT_CANDIDATES:
    if (candidate / "model").is_dir() and (candidate / "utils").is_dir():
        PROJECT_ROOT = candidate.resolve()
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError("Unable to locate the UNIReID project root.")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import build_lora_model, build_model
from reid5o_datasets.bases import GalleryDataset, tokenize
from reid5o_datasets.build import build_transforms
from reid5o_datasets.orbench import ORBench
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs, read_image
from utils.simple_tokenizer import SimpleTokenizer


MODALITY_ENCODERS = {
    "nir": "encode_nir_cls",
    "cp": "encode_cp_cls",
    "sk": "encode_sk_cls",
    "text": "encode_text_cls",
}

MODALITY_WEIGHT_INDEX = {
    "nir": 0,
    "cp": 1,
    "sk": 2,
    "text": 3,
}

MODALITY_TITLES = {
    "text": "TEXT",
    "nir": "IR",
    "sk": "SK",
    "cp": "CP",
}

QUERY_SPECS = {
    "text": {
        "dataset_key": "TEXT",
        "display_name": "TEXT",
        "modalities": ["text"],
        "positions": {"text": 1},
    },
    "text+ir": {
        "dataset_key": "TEXT+NIR",
        "display_name": "TEXT+IR",
        "modalities": ["text", "nir"],
        "positions": {"text": 1, "nir": 2},
    },
    "text+ir+sk": {
        "dataset_key": "TEXT+NIR+SK",
        "display_name": "TEXT+IR+SK",
        "modalities": ["text", "nir", "sk"],
        "positions": {"text": 1, "nir": 2, "sk": 3},
    },
    "text+ir+sk+cp": {
        # ORBench json uses TEXT+NIR+CP+SK; reorder it here for the requested view.
        "dataset_key": "TEXT+NIR+CP+SK",
        "display_name": "TEXT+IR+SK+CP",
        "modalities": ["text", "nir", "sk", "cp"],
        "positions": {"text": 1, "nir": 2, "cp": 3, "sk": 4},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ORBench retrieval results.")
    parser.add_argument(
        "--config_file",
        default="/mnt/18T/yinhao/results/logs/ORBench/20260414_171241_baseline/configs.yaml",
    )
    parser.add_argument(
        "--query_modes",
        nargs="+",
        default=["text", "text+ir", "text+ir+sk", "text+ir+sk+cp"],
    )
    parser.add_argument("--query_idx", type=int, default=7)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--figure_dir",
        default=str(SCRIPT_DIR / "retrieval_visualizations"),
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--use_weight", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def normalize_query_mode(query_mode):
    key = query_mode.lower().replace(" ", "")
    alias_map = {
        "text": "text",
        "text+nir": "text+ir",
        "text+ir": "text+ir",
        "text+nir+sk": "text+ir+sk",
        "text+ir+sk": "text+ir+sk",
        "text+nir+cp+sk": "text+ir+sk+cp",
        "text+nir+sk+cp": "text+ir+sk+cp",
        "text+ir+cp+sk": "text+ir+sk+cp",
        "text+ir+sk+cp": "text+ir+sk+cp",
    }
    if key not in alias_map:
        valid = ", ".join(QUERY_SPECS.keys())
        raise ValueError(f"Unsupported query mode '{query_mode}'. Valid modes: {valid}")
    return alias_map[key]


def resolve_path(path_str, base_dirs=None):
    path = Path(path_str).expanduser()
    candidates = [path]

    if not path.is_absolute():
        for base_dir in base_dirs or []:
            candidates.append(Path(base_dir).expanduser() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def resolve_checkpoint_path(cli_checkpoint, output_dir, config_file):
    base_dirs = [PROJECT_ROOT, Path.cwd(), Path(config_file).resolve().parent]
    resolved_output_dir = resolve_path(output_dir, base_dirs=base_dirs)

    if cli_checkpoint:
        checkpoint_path = resolve_path(
            cli_checkpoint,
            base_dirs=[resolved_output_dir, *base_dirs],
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    for name in ["R1_best.pth", "best.pth", "mAP_best.pth"]:
        candidate = resolved_output_dir / name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"No checkpoint found under {resolved_output_dir}. "
        "Tried: R1_best.pth, best.pth, mAP_best.pth"
    )


def build_gallery_loader(cfg, dataset):
    val_transforms = build_transforms(img_size=cfg.img_size, is_train=False)

    gallery_set = GalleryDataset(
        dataset.test["gallery_pids"],
        dataset.test["gallery_paths"],
        val_transforms,
    )

    gallery_loader = DataLoader(
        gallery_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return gallery_loader, val_transforms


def load_model(cfg, checkpoint_path, device):
    model = build_lora_model(cfg) if getattr(cfg, "add_lora", False) else build_model(cfg)

    checkpointer = Checkpointer(model)
    checkpointer.load(f=str(checkpoint_path))

    model.to(device)
    model.eval()

    return model


def extract_gallery_features(model, gallery_loader, device):
    gids, gfeats = [], []

    for pid, img in gallery_loader:
        img = img.to(device)

        with torch.no_grad():
            img_feat = model.encode_rgb_cls(img)

        gids.append(pid.view(-1))
        gfeats.append(img_feat)

    gids = torch.cat(gids, dim=0)
    gfeats = torch.cat(gfeats, dim=0)
    gfeats = F.normalize(gfeats, p=2, dim=1)

    return gids, gfeats


def get_weight_adapter(model, num_modalities, modality):
    adapter_name = {
        2: "mm_two_adapter",
        3: "mm_three_adapter",
        4: "mm_four_adapter",
    }.get(num_modalities)

    if adapter_name is None:
        raise ValueError(f"Unsupported number of modalities: {num_modalities}")

    adapter = getattr(model, adapter_name, None)

    if adapter is None:
        raise AttributeError(f"Model does not provide {adapter_name}")

    return adapter[MODALITY_WEIGHT_INDEX[modality]]


def fuse_query_features(model, features, modalities, use_weight):
    if len(features) == 1:
        return features[0], None

    if not use_weight:
        fused = torch.stack(features, dim=0).mean(dim=0)
        return fused, None

    weights = []

    for feature, modality in zip(features, modalities):
        adapter = get_weight_adapter(model, len(modalities), modality)
        weights.append(adapter(feature))

    weight_tensor = torch.cat(weights, dim=-1)
    norm_weight = F.softmax(weight_tensor, dim=-1)

    fused = 0
    for idx, feature in enumerate(features):
        fused = fused + feature * norm_weight[..., idx : idx + 1]

    return fused, norm_weight


def build_query_record(dataset, query_mode, query_idx):
    spec = QUERY_SPECS[query_mode]
    queries = dataset.test["queries"][spec["dataset_key"]]

    if query_idx < 0 or query_idx >= len(queries):
        raise IndexError(
            f"query_idx={query_idx} is out of range for {query_mode}. "
            f"Valid range: [0, {len(queries) - 1}]"
        )

    raw_query = queries[query_idx]

    record = {
        "mode_key": query_mode,
        "display_name": spec["display_name"],
        "query_idx": query_idx,
        "pid": raw_query[0],
        "modalities": spec["modalities"],
        "content": {},
    }

    for modality, position in spec["positions"].items():
        record["content"][modality] = raw_query[position]

    return record


def encode_query(model, query_record, transform, tokenizer, text_length, device, use_weight):
    features = []

    for modality in query_record["modalities"]:
        content = query_record["content"][modality]
        encoder = getattr(model, MODALITY_ENCODERS[modality])

        with torch.no_grad():
            if modality == "text":
                caption = tokenize(
                    content,
                    tokenizer=tokenizer,
                    text_length=text_length,
                    truncate=True,
                ).unsqueeze(0).to(device)

                feature = encoder(caption)

            else:
                image = read_image(content)
                image = transform(image).unsqueeze(0).to(device)

                feature = encoder(image)

        features.append(feature)

    fused_feature, weights = fuse_query_features(
        model,
        features,
        query_record["modalities"],
        use_weight=use_weight,
    )

    fused_feature = F.normalize(fused_feature, p=2, dim=1)

    if weights is None:
        return fused_feature, None

    return fused_feature, weights.squeeze(0).detach().cpu().tolist()


def retrieve_topk(qfeat, gfeats, topk):
    similarity = qfeat @ gfeats.t()

    scores, indices = torch.topk(
        similarity,
        k=min(topk, gfeats.size(0)),
        dim=1,
        largest=True,
        sorted=True,
    )

    return scores.squeeze(0).cpu(), indices.squeeze(0).cpu()


def build_pid_to_gallery_paths(dataset):
    mapping = defaultdict(list)

    for pid, path in zip(dataset.test["gallery_pids"], dataset.test["gallery_paths"]):
        mapping[int(pid)].append(path)

    return mapping


def set_axis_style(ax, border_color="#d0d0d0", title=None):
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)

    if title:
        ax.set_title(title, fontsize=11)


def plot_text_panel(ax, text, title=None):
    ax.set_facecolor("#f4f4f4")

    wrapped_text = fill(text, width=32)

    ax.text(
        0.06,
        0.90,
        wrapped_text,
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.25,
        transform=ax.transAxes,
    )

    set_axis_style(ax, border_color="#4c78a8", title=title)


def plot_image_panel(ax, image_path, title=None, border_color="#d0d0d0"):
    image = read_image(image_path)

    ax.imshow(image)
    set_axis_style(ax, border_color=border_color, title=title)


def build_weight_text(query_record, weights):
    if weights is None:
        return ""

    parts = []

    for modality, weight in zip(query_record["modalities"], weights):
        parts.append(f"{MODALITY_TITLES[modality]}={weight:.3f}")

    return " | weights: " + ", ".join(parts)


def save_visualization(
    query_record,
    gallery_paths,
    pid_to_gallery_paths,
    gids,
    scores,
    indices,
    topk,
    save_path,
    weights,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    query_save_path = save_path.with_name(save_path.stem + "_query.png")
    results_save_path = save_path.with_name(save_path.stem + "_results.png")

    # =========================
    # 1. 单独保存 Query
    # =========================
    query_panel_count = len(query_record["modalities"])

    fig_q, axes_q = plt.subplots(
        1,
        query_panel_count,
        figsize=(3.0 * query_panel_count, 3.6),
    )

    if query_panel_count == 1:
        axes_q = [axes_q]

    for ax in axes_q:
        ax.axis("off")

    for col, modality in enumerate(query_record["modalities"]):
        ax = axes_q[col]
        ax.axis("on")

        content = query_record["content"][modality]

        if modality == "text":
            plot_text_panel(
                ax,
                content,
                # MODALITY_TITLES[modality],
            )
        else:
            plot_image_panel(
                ax,
                content,
                # title=MODALITY_TITLES[modality],
                border_color="#4c78a8",
            )

    fig_q.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.90,
        bottom=0.02,
        wspace=0.05,
    )

    fig_q.savefig(
        query_save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig_q)

    # =========================
    # 2. 单独保存 Retrieval Results
    # =========================
    result_count = min(topk, len(indices))

    fig_r, axes_r = plt.subplots(
        1,
        result_count,
        figsize=(2.15 * result_count, 3.4),
    )

    if result_count == 1:
        axes_r = [axes_r]

    for ax in axes_r:
        ax.axis("off")

    for rank in range(result_count):
        ax = axes_r[rank]
        ax.axis("on")

        gallery_index = int(indices[rank])
        gallery_pid = int(gids[gallery_index])

        border_color = "#1f9d55" if gallery_pid == int(query_record["pid"]) else "#d62728"

        plot_image_panel(
            ax,
            gallery_paths[gallery_index],
            title=None,
            border_color=border_color,
        )

    # 控制查询结果之间的间距
    fig_r.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.995,
        bottom=0.005,
        wspace=0.03,
    )

    fig_r.savefig(
        results_save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig_r)

    print(f"Saved query visualization to {query_save_path}")
    print(f"Saved retrieval results visualization to {results_save_path}")


def main():
    cli_args = parse_args()
    cfg = load_train_configs(cli_args.config_file)
    cfg.training = False

    query_modes = [normalize_query_mode(mode) for mode in cli_args.query_modes]

    device = torch.device(cli_args.device)
    tokenizer = SimpleTokenizer()

    dataset = ORBench(root=cfg.root_dir)

    gallery_loader, val_transforms = build_gallery_loader(cfg, dataset)
    gallery_paths = dataset.test["gallery_paths"]
    pid_to_gallery_paths = build_pid_to_gallery_paths(dataset)

    checkpoint_path = resolve_checkpoint_path(
        cli_args.checkpoint,
        cfg.output_dir,
        cli_args.config_file,
    )

    model = load_model(cfg, checkpoint_path, device)

    gids, gfeats = extract_gallery_features(model, gallery_loader, device)

    use_weight = bool(cli_args.use_weight or getattr(cfg, "use_weight", False))
    figure_dir = Path(cli_args.figure_dir).expanduser().resolve()

    for query_mode in query_modes:
        query_record = build_query_record(
            dataset,
            query_mode,
            cli_args.query_idx,
        )

        qfeat, weights = encode_query(
            model,
            query_record,
            val_transforms,
            tokenizer,
            cfg.text_length,
            device,
            use_weight=use_weight,
        )

        scores, indices = retrieve_topk(
            qfeat,
            gfeats,
            cli_args.topk,
        )

        save_path = figure_dir / f"{query_mode.replace('+', '_')}_idx{cli_args.query_idx}.png"

        save_visualization(
            query_record,
            gallery_paths,
            pid_to_gallery_paths,
            gids,
            scores,
            indices,
            cli_args.topk,
            save_path,
            weights,
        )

        print(f"Saved {query_record['display_name']} visualization to {save_path}")


if __name__ == "__main__":
    main()