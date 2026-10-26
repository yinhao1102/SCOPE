import os
import os.path as op
import torch
import numpy as np
import random
import time

from reid5o_datasets import build_reid5o_dataloader
from tri_datasets import build_tri_dataloader, is_tri_dataset
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_lora_model
from utils.evaluation import build_orbench_evaluator, build_tri_evaluator
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
    logger = setup_logger('SCOPE', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    # logger.info("Training only sketch {}".format(args.only_sketch))
    # logger.info("Using only text {}".format(args.only_text))
    # logger.info("Using only fusion {}".format(args.only_fusion_loss))
    # logger.info("Using {} fusion method".format(args.fusion_way))

    save_train_configs(args.output_dir, args)

    if is_tri_dataset(args.dataset_name):
        dataloaders = build_tri_dataloader(args)
    elif args.dataset_name == 'ORBench':
        dataloaders = build_reid5o_dataloader(args)
    else:
        raise ValueError(
            "dataset_name must be ORBench, Tri-CUHK-PEDES, or Tri-RSTPReid"
        )
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
    if is_tri_dataset(args.dataset_name):
        evaluator = build_tri_evaluator(dataloaders, training=True)
    else:
        evaluator = build_orbench_evaluator(dataloaders, training=True)

    start_epoch = 1

    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
