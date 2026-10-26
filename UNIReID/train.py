import os
import os.path as op
import torch
import numpy as np
import random
import time

# from datasets import build_dataloader
from model.mmencoder_with_ada_dual_lora import AdaptiveDualRankAllocator
from reid5o_datasets import build_reid5o_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model, build_lora_model
from utils.metrics import Evaluator
from utils.reid5o_metrics import Evaluator1
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_trainable_modules(model):
    print("=" * 120)
    print("Trainable modules:")
    printed = set()

    for name, p in model.named_parameters():
        if p.requires_grad:
            module_name = name.rsplit(".", 1)[0] if "." in name else name
            if module_name not in printed:
                printed.add(module_name)
                print(module_name)

    print("=" * 120)
    print(f"Number of trainable modules: {len(printed)}")
    print("=" * 120)

if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    # args.log_dir = op.join(args.output_dir, args.log_dir)
    logger = setup_logger('UNIReid_lora', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    # logger.info("Training only sketch {}".format(args.only_sketch))
    # logger.info("Using only text {}".format(args.only_text))
    # logger.info("Using only fusion {}".format(args.only_fusion_loss))
    # logger.info("Using {} fusion method".format(args.fusion_way))

    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    # train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
    dataloaders = build_reid5o_dataloader(args)
    train_loader = dataloaders[0]
    num_classes = dataloaders[-1]
    model = build_lora_model(args, num_classes)
    # print(model)
    model.to(device)

    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params} ({total_params / 1e6:.3f} M)")
    print(f"Trainable params: {trainable_params} ({trainable_params / 1e6:.3f} M)")
    # print(f"Non-trainable params: {non_trainable_params} ({non_trainable_params / 1e6:.3f} M)")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")

    # print_trainable_modules(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    # evaluator = Evaluator(args, val_img_loader, val_txt_loader, val_sketch_loader)

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
    text_nir_cp_sk_query_loader = dataloaders[1:-1]
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

    start_epoch = 1

    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)