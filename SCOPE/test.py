import argparse
import logging
import os.path as osp

import torch

from model import build_lora_model
from reid5o_datasets import build_reid5o_dataloader
from tri_datasets import build_tri_dataloader, is_tri_dataset
from utils.checkpoint import Checkpointer
from utils.evaluation import build_orbench_evaluator, build_tri_evaluator
from utils.iotools import load_train_configs
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCOPE")
    parser.add_argument("--config_file", required=True, help="path to the training configs.yaml")
    parser.add_argument("--checkpoint", default="R1_best.pth", help="checkpoint name or path")
    parser.add_argument("--root_dir", default=None, help="override the dataset root in configs.yaml")
    parser.add_argument("--no_msf", action="store_true", help="use average fusion instead of MSF")
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    args = load_train_configs(cli_args.config_file)
    run_dir = osp.dirname(osp.abspath(cli_args.config_file))
    args.output_dir = run_dir
    if cli_args.root_dir is not None:
        args.root_dir = cli_args.root_dir
    args.training = False

    logger = setup_logger('SCOPE.test', save_dir=args.output_dir, if_train=False)
    logger.info(args)

    if is_tri_dataset(args.dataset_name):
        dataloaders = build_tri_dataloader(args)
        evaluator = build_tri_evaluator(dataloaders, training=False)
    elif args.dataset_name == 'ORBench':
        dataloaders = build_reid5o_dataloader(args)
        evaluator = build_orbench_evaluator(dataloaders, training=False)
    else:
        raise ValueError(
            "dataset_name must be ORBench, Tri-CUHK-PEDES, or Tri-RSTPReid"
        )

    model = build_lora_model(args, num_classes=dataloaders[-1])
    checkpoint_path = cli_args.checkpoint
    if not osp.isabs(checkpoint_path):
        checkpoint_path = osp.join(run_dir, checkpoint_path)
    Checkpointer(model).load(f=checkpoint_path)
    model.to("cuda").eval()

    logger = logging.getLogger("SCOPE.test")
    logger.info("Evaluating %s with MSF=%s", args.dataset_name, not cli_args.no_msf)
    top1, mean_ap = evaluator.eval(model, use_weight=not cli_args.no_msf)
    logger.info("Average R1: %.4f, average mAP: %.4f", top1, mean_ap)
