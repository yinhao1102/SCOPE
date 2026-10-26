import logging
from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import torch.nn.parallel
import numpy as np
import time
import os.path as op

# import clip
from datasets import build_dataloader
from reid5o_datasets import build_reid5o_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from model import build_lora_model
from utils.metrics import Evaluator
from utils.reid5o_metrics import Evaluator1
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    parser.add_argument("--config_file", default='/mnt/18T/yinhao/results/logs/ORBench/20260414_171241_baseline/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
 

    args.training = True
    logger = setup_logger('CLIP2ReID', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    
    # test_img_loader, test_txt_loader, test_sketch_loader = build_dataloader(args)
    train_loader, \
    test_gallery_loader, \
    nir_query_loader, \
    cp_query_loader, \
    sk_query_loader, \
    text_query_loader, \
    nir_cp_query_loader, \
    cp_nir_query_loader, \
    nir_sk_query_loader, \
    sk_nir_query_loader, \
    nir_text_query_loader, \
    text_nir_query_loader, \
    cp_sk_query_loader, \
    sk_cp_query_loader, \
    cp_text_query_loader, \
    text_cp_query_loader, \
    sk_text_query_loader, \
    text_sk_query_loader, \
    nir_cp_sk_query_loader, \
    cp_nir_sk_query_loader, \
    sk_nir_cp_query_loader, \
    nir_cp_text_query_loader, \
    cp_nir_text_query_loader, \
    text_nir_cp_query_loader, \
    nir_sk_text_query_loader, \
    sk_nir_text_query_loader, \
    text_nir_sk_query_loader, \
    cp_sk_text_query_loader, \
    sk_cp_text_query_loader, \
    text_cp_sk_query_loader, \
    nir_cp_sk_text_query_loader, \
    cp_nir_sk_text_query_loader, \
    sk_nir_cp_text_query_loader, \
    text_nir_cp_sk_query_loader, num_classes = build_reid5o_dataloader(args)
    model = build_lora_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'R1_best.pth'))
    model.to(device)
    
    # do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader)
        # 初始化 Evaluator
    evaluator = Evaluator1(
        gallery_loader=test_gallery_loader,
        get_mAP=True,
        # 单模态
        nir_query_loader=nir_query_loader,
        cp_query_loader=cp_query_loader,
        sk_query_loader=sk_query_loader,
        text_query_loader=text_query_loader,
        # 双模态
        nir_cp_query_loader=nir_cp_query_loader,
        cp_nir_query_loader=cp_nir_query_loader,
        nir_sk_query_loader=nir_sk_query_loader,
        sk_nir_query_loader=sk_nir_query_loader,
        nir_text_query_loader=nir_text_query_loader,
        text_nir_query_loader=text_nir_query_loader,
        cp_sk_query_loader=cp_sk_query_loader,
        sk_cp_query_loader=sk_cp_query_loader,
        cp_text_query_loader=cp_text_query_loader,
        text_cp_query_loader=text_cp_query_loader,
        sk_text_query_loader=sk_text_query_loader,
        text_sk_query_loader=text_sk_query_loader,
        # 三模态
        nir_cp_sk_query_loader=nir_cp_sk_query_loader,
        cp_nir_sk_query_loader=cp_nir_sk_query_loader,
        sk_nir_cp_query_loader=sk_nir_cp_query_loader,
        nir_cp_text_query_loader=nir_cp_text_query_loader,
        cp_nir_text_query_loader=cp_nir_text_query_loader,
        text_nir_cp_query_loader=text_nir_cp_query_loader,
        nir_sk_text_query_loader=nir_sk_text_query_loader,
        sk_nir_text_query_loader=sk_nir_text_query_loader,
        text_nir_sk_query_loader=text_nir_sk_query_loader,
        cp_sk_text_query_loader=cp_sk_text_query_loader,
        sk_cp_text_query_loader=sk_cp_text_query_loader,
        text_cp_sk_query_loader=text_cp_sk_query_loader,
        # 四模态
        nir_cp_sk_text_query_loader=nir_cp_sk_text_query_loader,
        cp_nir_sk_text_query_loader=cp_nir_sk_text_query_loader,
        sk_nir_cp_text_query_loader=sk_nir_cp_text_query_loader,
        text_nir_cp_sk_query_loader=text_nir_cp_sk_query_loader
    )


    logger = logging.getLogger("ORBench.test")
    logger.info("Enter evaluating")

    model.eval()
    # model.vision_encoder.transformer.visualize_average_lora_gates(
    # save_dir="./figures/lora_gates",
    # lora_indices=[0, 1, 2, 3],
    # )
    # model.vision_encoder.transformer.visualize_single_modal_lora_gates(
    #     save_dir="./figures/lora_gates",
    #     lora_indices=[0, 1, 2, 3],
    #     lora_name_map={
    #         0: "RGB",
    #         1: "IR",
    #         2: "SK",
    #         3: "CP",
    #     },
    #     layer_name="c_proj",
    # )
    model.visualize_two_modal_contribution(
        data_loader=train_loader,
        save_path="./figures/two_modal_contribution.png",
        temperature=args.adapter_temperature,
        combos=[
            (3, 0),  # Text + IR
            (3, 2),  # Text + SK
            (3, 1),  # Text + CP
        ],
        max_batches=1,
    )
    top1 = evaluator.eval(model.eval())
