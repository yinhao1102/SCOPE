"""One-command HAM initialization and SCOPE forward check."""
import argparse
import itertools
import json

import torch

from . import SCOPE
from .model import attach_msf, fusion_mode, load_heads


def main():
    parser = argparse.ArgumentParser(description="Load HAM weights and check the SCOPE model")
    parser.add_argument("--ham-checkpoint", default="pretrained/ham.pth")
    parser.add_argument("--encoder-checkpoint", help="Optional trained SCOPE encoder state")
    parser.add_argument("--fusion-checkpoint", help="Optional trained MSF head state")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rank", type=int, default=8, help="Rank of each Dual-LoRA branch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.rank, args.batch_size, args.threads) < 1:
        parser.error("Rank, batch size, and thread count must be positive")
    if args.fusion_checkpoint and not args.encoder_checkpoint:
        parser.error("--fusion-checkpoint requires its matching --encoder-checkpoint")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2)
    print("Loading HAM pretrained weights...", flush=True)
    model = SCOPE.from_pretrained(args.ham_checkpoint,
                                  rank_magnitude=args.rank, rank_direction=args.rank)
    if args.encoder_checkpoint:
        state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)
    model.to(device).eval()
    attach_msf(model)
    if args.fusion_checkpoint:
        state = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=True)
        load_heads(model, state)
    fusion_mode(model, training=False)
    width = model.clip_text_encoder.token_embedding.num_embeddings
    length = model.clip_text_encoder.positional_embedding.shape[0]
    tokens = torch.zeros(args.batch_size, length, dtype=torch.long, device=device)
    tokens[:, 0], tokens[:, 1], tokens[:, 2] = width - 2, 1, width - 1
    inputs = {name: torch.randn(args.batch_size, 3, 384, 128, device=device)
              for name in ("rgb", "nir", "cp", "sk")}
    inputs["text"] = tokens
    print("Running synthetic model inputs; this check does not measure retrieval accuracy.", flush=True)
    with torch.no_grad():
        features = model(inputs)
        if not all(torch.isfinite(value).all() for value in features.values()):
            raise RuntimeError("Nonfinite encoder output")
        modalities = [features[name] for name in ("nir", "cp", "sk", "text")]
        for size in (2, 3, 4):
            for indices in itertools.combinations(range(4), size):
                fused, _, weights = model.msf_fuse_by_indices(modalities, indices)
                if not torch.isfinite(fused).all():
                    raise RuntimeError("Nonfinite fusion output")
                torch.testing.assert_close(weights.sum(1), torch.ones(args.batch_size, device=device))
    print(json.dumps({"status": "passed", "backbone": "HAM", "device": device,
                      "encoder_features": {key: list(value.shape) for key, value in features.items()},
                      "fusion_subsets_checked": 11,
                      "msf_weights": "loaded" if args.fusion_checkpoint else "newly initialized"}, indent=2))


if __name__ == "__main__":
    main()
