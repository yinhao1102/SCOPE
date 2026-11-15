"""Synthetic files exercise the real loaders, two-stage trainer, and evaluator."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from SCOPE.datasets.build import DataBundle
from SCOPE.datasets.tri import TRI_DATASETS
from SCOPE.model.ham import HAMBackbone
from SCOPE.processor.processor import train_main, evaluate_main
from SCOPE.utils.evaluation import ORBENCH_QUERY_LOADERS
from SCOPE.utils.options import arguments


def write_image(path, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (32, 16, 3), dtype=np.uint8)).save(path)


def fixture(root):
    rng = np.random.default_rng(41)
    for name, spec in TRI_DATASETS.items():
        directory = root / spec["directory"]
        rows = []
        for split in ("train", "val", "test"):
            for pid in (1, 2, 3):
                for view in (0, 1):
                    relative = f"{split}/{pid}_{view}.png"
                    write_image(directory / spec["image_directory"] / relative, rng)
                    write_image(directory / spec["sketch_directory"] / relative, rng)
                    rows.append({"split": split, "id": pid, spec["path_key"]: relative,
                                 "captions": ["person wearing black", "person in a coat"]})
        # Exercise the commonly distributed underscore spelling for ICFG.
        annotation = "ICFG_PEDES.json" if name == "Tri-ICFG-PEDES" else spec["annotation"]
        (directory / annotation).write_text(json.dumps(rows))
    directory = root / "ORBench"
    rows, gallery = [], []
    for pid in (1, 2, 3):
        for view in (0, 1):
            for modality in ("vis", "nir", "cp", "sk"):
                write_image(directory / modality / str(pid) / f"{view}.png", rng)
            path = f"vis/{pid}/{view}.png"
            rows.append({"id": pid, "file_path": path, "caption": "person wearing black"})
            gallery.append([pid, path])
    queries = {"RGB_GALLERY": gallery}
    for loader in ORBENCH_QUERY_LOADERS:
        modalities = loader.removesuffix("_query_loader").split("_")
        queries["+".join(modalities).upper()] = [
            [pid, *("person in photo.jpg wearing a coat" if m == "text"
                    else f"{m}/{pid}/0.png" for m in modalities)] for pid in (1, 2, 3)
        ]
    (directory / "train_annos.json").write_text(json.dumps(rows))
    (directory / "test_gallery_and_queries.json").write_text(json.dumps(queries))


def tiny_ham(path):
    shape = {
        "visual.conv1.weight": torch.empty(64, 3, 16, 16),
        "text_projection": torch.empty(64, 32),
        "token_embedding.weight": torch.empty(49408, 64),
        "positional_embedding": torch.empty(77, 64),
        "visual.transformer.resblocks.0.attn.in_proj_weight": torch.empty(192, 64),
        "transformer.resblocks.0.attn.in_proj_weight": torch.empty(192, 64),
    }
    torch.manual_seed(7)
    model = HAMBackbone(shape, (32, 16))
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.normal_(0, 0.02)
            if "ln_" in name:
                value.fill_(1 if name.endswith("weight") else 0)
    torch.save({"model": {"module.base_model." + k: v for k, v in model.state_dict().items()}}, path)


class TrainingPipelineTest(unittest.TestCase):
    def test_all_datasets_train_reload_export_and_freeze(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "datasets")
            tiny_ham(root / "ham.pth")
            for dataset in ("ORBench", *TRI_DATASETS):
                with self.subTest(dataset=dataset):
                    output = root / dataset
                    config = root / (dataset + ".yaml")
                    selection = "val" if dataset == "Tri-RSTPReid" else "test"
                    config.write_text(yaml.safe_dump(dict(
                        dataset_name=dataset, root_dir=str(root / "datasets"),
                        pretrain_choice=str(root / "ham.pth"), output_dir=str(output),
                        device="cpu", num_workers=0, fusion_workers=0, threads=1,
                        img_size=[32, 16], batch_size=3, test_batch_size=3,
                        encoder_epochs=2, fusion_epochs=2, mca_probe_batches=2,
                        mca_bank_identities=3, amp_dtype="none", selection_split=selection,
                    )))
                    args = arguments(argv=["--config", str(config)])
                    bundle = DataBundle(args)
                    self.assertEqual(bundle.check_data()["status"], "passed")
                    del bundle
                    with patch("sys.argv", ["train.py", "--config", str(config)]), contextlib.redirect_stdout(io.StringIO()):
                        train_main()
                    for stage in (1, 2):
                        directory = output / f"stage{stage}"
                        best = json.loads((directory / "best.json").read_text())
                        history = [json.loads(line) for line in
                                   (directory / f"metrics_{selection}.jsonl").read_text().splitlines()]
                        self.assertEqual(len(history), 2)
                        self.assertEqual(best["mean_mAP"], max(row["mean_mAP"] for row in history))
                        checkpoint = torch.load(directory / "mAP_best.pth", weights_only=True)
                        self.assertEqual(checkpoint["epoch"], best["epoch"])
                        self.assertEqual(checkpoint["metrics"]["per_query"], best["per_query"])
                        with (directory / "best_per_query.csv").open() as stream:
                            csv_rows = list(csv.DictReader(stream))
                        self.assertEqual(len(csv_rows), 32 if dataset == "ORBench" else 4)
                        for row in csv_rows:
                            for metric in ("R1", "R5", "R10", "mAP"):
                                self.assertEqual(float(row[metric]), best["per_query"][row["setting"]][metric])
                    frozen = json.loads((output / "stage2/freeze_verified.json").read_text())
                    self.assertTrue(frozen["single_modal_metrics_unchanged"])
                    active = json.loads((output / "stage2/first_batch.json").read_text())["active_names"]
                    self.assertTrue(all(name.startswith("mm_") for name in active))
                    stage1 = json.loads((output / "stage1/transition.json").read_text())
                    stage2 = json.loads((output / "stage2/best.json").read_text())
                    self.assertEqual(stage1["per_group"]["ONE_AVER"], stage2["per_group"]["ONE_AVER"])
                    transition = torch.load(output / "stage1/transition.pth", weights_only=True)
                    fusion = torch.load(output / "stage2/mAP_best.pth", weights_only=True)
                    self.assertEqual(transition["epoch"], 2)
                    self.assertEqual(fusion["encoder_epoch"], 2)
                    self.assertEqual(fusion["encoder_sha256"], transition["encoder_sha256"])
                    self.assertEqual(fusion["global_epoch"], 2 + fusion["epoch"])
                    with patch("sys.argv", ["test.py", "--config", str(config), "--export-rankings", "--top-k", "3"]), contextlib.redirect_stdout(io.StringIO()):
                        evaluate_main()
                    evaluated = json.loads((output / "evaluation.json").read_text())
                    tested = [json.loads(line) for line in (output / "stage2/metrics_test.jsonl").read_text().splitlines()]
                    selected = next(row for row in tested if row["epoch"] == stage2["epoch"])
                    self.assertEqual(evaluated["per_query"], selected["per_query"])
                    retrievals = [json.loads(line) for line in (output / "rankings.jsonl").read_text().splitlines()]
                    self.assertTrue(retrievals)
                    self.assertTrue(all(len(row["gallery"]) == 3 for row in retrievals))
                    self.assertTrue(all(not Path(hit["path"]).is_absolute()
                                        for row in retrievals for hit in row["gallery"]))
                    if dataset != "ORBench":
                        # Generated annotations are ordered by PID and then view.
                        for row in retrievals:
                            if row["setting"] == "TEXT_original":
                                continue
                            source = row["query_index"] if row["setting"] == "SK" else row["query_index"] // 2
                            filename = f"{row['query_pid']}_{source % 2}.png"
                            self.assertTrue(all(Path(hit["path"]).name != filename for hit in row["gallery"]))

    def test_switch_uses_epoch_n_even_if_an_earlier_epoch_scores_better(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "datasets")
            tiny_ham(root / "ham.pth")
            output = root / "run"
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(dict(
                dataset_name="Tri-RSTPReid", root_dir=str(root / "datasets"),
                pretrain_choice=str(root / "ham.pth"), output_dir=str(output),
                device="cpu", num_workers=0, fusion_workers=0, threads=1,
                img_size=[32, 16], batch_size=3, test_batch_size=3,
                encoder_epochs=9, fusion_epochs=2, mca_probe_batches=2,
                mca_bank_identities=3, amp_dtype="none", selection_split="test",
            )))
            native_evaluate = DataBundle.evaluate
            encoder_evaluations = 0

            def earlier_epoch_scores_better(bundle, model, cache, fused=False):
                nonlocal encoder_evaluations
                result = native_evaluate(bundle, model, cache, fused)
                if not fused:
                    # Control only checkpoint-selection scores. Inputs, gradients,
                    # per-query evaluation and stage-2 training remain real.
                    encoder_evaluations += 1
                    result["mean_mAP"] = 100.0 - encoder_evaluations
                return result

            command = ["train.py", "--config", str(config), "--switch-epoch", "2"]
            with patch.object(DataBundle, "evaluate", earlier_epoch_scores_better), \
                    patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
                train_main()
            best = torch.load(output / "stage1/mAP_best.pth", weights_only=True)
            transition = torch.load(output / "stage1/transition.pth", weights_only=True)
            fusion = torch.load(output / "stage2/mAP_best.pth", weights_only=True)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(transition["epoch"], 2)
            self.assertGreater(best["metrics"]["mean_mAP"], transition["metrics"]["mean_mAP"])
            self.assertTrue(any(not torch.equal(value, best["model"][name])
                                for name, value in transition["model"].items()))
            self.assertEqual(fusion["encoder_sha256"], transition["encoder_sha256"])
            self.assertEqual(Path(fusion["encoder_checkpoint"]), output / "stage1/transition.pth")
            completion = json.loads((output / "stage1/completed.json").read_text())
            self.assertFalse(completion["best_checkpoint_reloaded"])

            def evaluate(*options, destination=output):
                with patch("sys.argv", ["test.py", "--config", str(config), "--output", str(destination), *options]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    evaluate_main()
                return json.loads((destination / "evaluation.json").read_text())

            result = evaluate()
            self.assertEqual(result["per_query"], fusion["metrics"]["per_query"])
            mean_result = evaluate("--no-fusion")
            self.assertEqual(mean_result["per_query"], transition["metrics"]["per_query"])
            with self.assertRaisesRegex(ValueError, "different encoder"):
                evaluate("--encoder-checkpoint", str(output / "stage1/mAP_best.pth"))

            # A separate MSF run must use the requested epoch as well. Its
            # evaluator resolves the recorded encoder outside its output folder.
            separate = root / "fusion_only"
            separate_command = command + ["--stage", "fusion", "--output", str(separate),
                                           "--encoder-checkpoint", str(output / "stage1/transition.pth")]
            with patch("sys.argv", separate_command), contextlib.redirect_stdout(io.StringIO()):
                train_main()
            separate_fusion = torch.load(separate / "stage2/mAP_best.pth", weights_only=True)
            self.assertEqual(separate_fusion["encoder_sha256"], transition["encoder_sha256"])
            self.assertEqual(evaluate(destination=separate)["per_query"], separate_fusion["metrics"]["per_query"])
            with patch("sys.argv", command + ["--stage", "fusion", "--encoder-checkpoint",
                                               str(output / "stage1/mAP_best.pth")]), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "requires encoder epoch 2"):
                train_main()

    def test_switch_argument_keeps_encoder_epochs_alias_and_rejects_zero(self):
        self.assertEqual(arguments(argv=["--switch-epoch", "21"]).encoder_epochs, 21)
        self.assertEqual(arguments(argv=["--encoder-epochs", "21"]).encoder_epochs, 21)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(argv=["--switch-epoch", "0"])


if __name__ == "__main__":
    unittest.main()
