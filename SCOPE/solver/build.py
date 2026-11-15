# MIT License for the portions adapted from the credited upstream projects.
#
# Copyright (c) 2023 anosorae
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "classifier" in key or "mcm_head" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        raise NotImplementedError(f"Unknown optimizer: {args.optimizer}")

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
