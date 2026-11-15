# SCOPE

Code for **SCOPE: A Contribution-aware and Efficient Framework for Multimodal Person Retrieval**, using the **HAM-pretrained ViT-B/16** backbone.

The repository provides data loading, image preprocessing and text tokenization, two-stage training, checkpoint selection, and retrieval evaluation. It supports ORBench, Tri-CUHK-PEDES, Tri-RSTPReid, and Tri-ICFG-PEDES through a common `train.py` / `test.py` interface.

## Installation

Requires Python 3.10+ and PyTorch 2.5+. Install a PyTorch build compatible with your CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Each training run uses one GPU. Set `CUDA_VISIBLE_DEVICES` to choose it. CPU execution is supported for small integration checks.

## HAM pretrained weights

Download the **ReID pretrained model** linked in the [official HAM README](https://github.com/MPI-Lab/HAM/blob/main/readme.md) and place it at `pretrained/ham.pth`, or specify another local file with `--ham-checkpoint`. The loader accepts the original HAM checkpoint and a backbone tensor state dictionary. It adapts the visual positional embeddings to the configured image resolution.

## Data preparation

Obtain the benchmark images, sketches, and annotations from their respective providers. `--data-root` points to the **parent** of the dataset folders:

```text
datasets/
├── ORBench/
│   ├── train_annos.json
│   ├── test_gallery_and_queries.json
│   ├── vis/<identity>/<image>
│   ├── nir/<identity>/<image>
│   ├── cp/<identity>/<image>
│   └── sk/<identity>/<image>
├── CUHK-PEDES/
│   ├── reid_raw.json
│   ├── imgs/<relative image path>
│   └── imgs-sketch2/<same relative image path>
├── RSTPReid/
│   ├── data_captions.json
│   ├── imgs/<relative image path>
│   └── img-sketch2/<same relative image path>
└── ICFG-PEDES/
    ├── ICFG-PEDES.json
    ├── imgs/<relative image path>
    └── imgs-sketch/<same relative image path>
```

Only the dataset selected for a run is required. Both training and evaluation use its complete benchmark annotations. Keep the official image paths and split assignments. For the Tri datasets, each sketch must have the same relative path as its corresponding RGB image; RGB images and captions alone are insufficient. `ICFG_PEDES.json` is also accepted for ICFG.

ORBench training records contain `id`, `file_path`, and `caption`; the visual query images are sampled from the same identity. Its test annotations contain `RGB_GALLERY` and all 32 native query settings, with rows ordered as `[identity, query_1, ...]`. Tri annotations are lists of records with `id`, `split`, `captions`, and `file_path` (`img_path` for RSTPReid).

Images are resized to 384 × 128 and normalized with CLIP statistics. Training uses random horizontal flips. ORBench retains the training caption masking used by the original loader; Tri training and all evaluation use unmasked captions. Text is tokenized to 77 tokens. The public CLIP BPE vocabulary is bundled in `SCOPE/data/`.

Check the referenced image files and a collated input sample before training; this does not load HAM:

```bash
python train.py --config configs/ham_orbench.yaml \
  --data-root datasets --check-data
```

## Training

Run both stages with one command:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh train \
  --config configs/ham_orbench.yaml \
  --data-root datasets \
  --ham-checkpoint pretrained/ham.pth \
  --output outputs/ham_orbench
```

The first stage trains the free-gate Dual-LoRA encoders and modality tokenizers with single-modal SDM alignment. It evaluates after every epoch and selects the best encoder by Avg-mAP. The second stage loads that encoder, freezes it, and trains only MSF with multimodal SDM + **1.5 MCA**. MCA targets and temperature calibration use training samples and a training RGB reference bank. The encoder remains in evaluation mode throughout the second stage.

The defaults use ranks 8 + 8, batch size 64, 60 encoder epochs, and 35 fusion epochs. YAML files expose the learning rates, epoch budgets, ranks, MCA weight, and other settings. CUDA training uses BF16 autocast; set `amp_dtype: none` in the configuration for FP32. Use a new output directory for each experiment; existing training histories are not overwritten.

| Dataset | Configuration | Query modalities |
|---|---|---|
| ORBench | `configs/ham_orbench.yaml` | NIR, CP, SK, Text and their combinations |
| Tri-CUHK-PEDES | `configs/ham_cuhk.yaml` | SK, Text, SK + Text |
| Tri-RSTPReid | `configs/ham_rstp.yaml` | SK, Text, SK + Text |
| Tri-ICFG-PEDES | `configs/ham_icfg.yaml` | SK, Text, SK + Text |

For example:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh train \
  --config configs/ham_cuhk.yaml --data-root datasets \
  --ham-checkpoint pretrained/ham.pth --output outputs/ham_cuhk
```

The stages can also be run separately:

```bash
python train.py --config configs/ham_orbench.yaml --stage encoder

python train.py --config configs/ham_orbench.yaml --stage fusion \
  --encoder-checkpoint outputs/ham_orbench/stage1/mAP_best.pth
```

Use `--rank 8`, `--mca-weight 1.5`, `--workers 4`, or `--batch-size 64` for common overrides. `--dry-run` prints the resolved configuration. `--mca-weight 0` trains MSF without MCA. These entry points start a new stage rather than resuming an interrupted optimizer state.

## Evaluation and outputs

Evaluate the selected encoder and MSF checkpoints:

```bash
bash run.sh test --config configs/ham_orbench.yaml \
  --data-root datasets --ham-checkpoint pretrained/ham.pth \
  --output outputs/ham_orbench --export-rankings --top-k 10
```

By default, evaluation loads `stage1/mAP_best.pth` and, when present, `stage2/mAP_best.pth` from the output directory. To evaluate another run, pass `--encoder-checkpoint` and `--fusion-checkpoint`. Use `--no-fusion` to evaluate the encoder with arithmetic mean fusion. MSF checkpoints are checked against their encoder fingerprint to prevent mixing stages from different runs.

For a custom training configuration, evaluate with the saved configuration so that ranks and preprocessing match:

```bash
python test.py --config outputs/ham_orbench/stage1/configs.yaml \
  --export-rankings
```

The output directory contains:

```text
outputs/ham_orbench/
├── train.log
├── stage1/
│   ├── configs.yaml
│   ├── parameter_counts.json
│   ├── metrics_test.jsonl
│   ├── mAP_best.pth
│   ├── R1_best.pth
│   ├── best.json
│   ├── best_per_query.csv
│   └── best_per_group.csv
├── stage2/
│   ├── ...                         # same metric/checkpoint files
│   ├── calibration.json
│   └── freeze_verified.json
├── evaluation.json
├── evaluation_per_query.csv
├── evaluation_per_group.csv
└── rankings.jsonl                  # optional top-k gallery paths and scores
```

Evaluation reports **R@1, R@5, R@10, mAP, and mINP** in percent for each query setting and search group. `rankings.jsonl` identifies the query setting, annotation row index, identity, and retrieved gallery images with cosine scores. Exported image paths are relative to the dataset root.

### Evaluation protocol

- **ORBench:** all 32 native query settings are retained, including the annotation-specific modality orderings. Single-, dual-, tri-, and quad-modal groups contain 4, 12, 12, and 4 settings. Avg-mAP gives equal weight to the four group means.
- **Tri datasets:** the primary TEXT, SK, and TEXT + SK evaluations exclude the query's paired source RGB image from the gallery. Single-modal metrics average TEXT and SK; dual-modal metrics use TEXT + SK. Checkpoint selection averages the three query mAP values. `TEXT_original` additionally reports text retrieval without source-image exclusion and is not included in this average.
- Presets retain the experimental setting `selection_split: test`. These are best-test-selected results, not scores on an independently held-out test set. Where the Tri annotations provide a validation split, use `--selection-split val` for validation-based checkpoint selection, then `test.py --split test` for evaluation. ORBench's native annotations provide train/test only.
- Every metric in a best-result table comes from the same selected checkpoint. During MSF training, single-modal scores remain fixed because the encoder is frozen. The best checkpoint is reloaded and its metrics verified at the end of each stage.

## Code structure

```text
train.py / test.py           Training and retrieval evaluation entry points
configs/                    HAM dataset presets and shared defaults
SCOPE/datasets/             Benchmark parsing, transforms, and data loaders
SCOPE/model/ham.py          HAM checkpoint loading and backbone construction
SCOPE/model/scope.py        Modality encoders and backbone integration
SCOPE/model/gd_lora.py      Free-gate Dual-LoRA
SCOPE/model/msf.py          Modality-specific fusion heads
SCOPE/model/mca.py          Modality contribution alignment
SCOPE/model/objectives.py   Similarity distribution matching
SCOPE/processor/            Training stages, checkpointing, and result exports
SCOPE/solver/               Optimizer and learning-rate scheduling
SCOPE/utils/                Tokenization and retrieval metrics
tests/                      Generated-data integration and evaluator checks
```

The free gate uses AB-norm rank reweighting, learned positive strength, and output-channel RMS calibration, without an additive direction residual. MSF combines raw modality features before final cosine normalization.

## Checks

Run a forward check with a local HAM checkpoint:

```bash
bash run.sh check --ham-checkpoint pretrained/ham.pth
```

Run CPU integration checks without downloading data or pretrained weights:

```bash
python -m unittest discover -s tests -v
```

The tests generate temporary images and a tiny backbone. They exercise both training stages, best-checkpoint reload, frozen-encoder consistency, metric and ranking exports, the four dataset formats, and equivalence of cached ORBench evaluation to the native evaluator. They verify the pipeline, not benchmark performance.

## Acknowledgements

We thank the authors of [IRRA](https://github.com/anosorae/IRRA), [HAM](https://github.com/MPI-Lab/HAM), [CLIP](https://github.com/openai/CLIP), and [UNIReID](https://github.com/ccq195/unireid) for making their research and code available. SCOPE uses the HAM-pretrained backbone, builds on the CLIP architecture and tokenizer, adopts IRRA's SDM objective and training-code conventions, and follows the Tri retrieval protocol used by UNIReID.
