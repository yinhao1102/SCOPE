import logging
import time
import torch
from utils.meter import AverageMeter
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    arguments2 = {}
    arguments2["num_epoch"] = num_epoch
    arguments2["iteration"] = 0

    arguments3 = {}
    arguments3["num_epoch"] = num_epoch
    arguments3["iteration"] = 0

    arguments4 = {}
    arguments4["num_epoch"] = num_epoch
    arguments4["iteration"] = 0

    logger = logging.getLogger("SCOPE.train")
    logger.info('start training')

    loss_meter = AverageMeter()
    ctwo_mca_loss_meter = AverageMeter()
    cthree_mca_loss_meter = AverageMeter()
    cfour_mca_loss_meter = AverageMeter()
    mcm_loss_meter = AverageMeter()
    mlm_loss_meter = AverageMeter()
    mlm_acc_meter = AverageMeter()
    text_itc_loss_meter = AverageMeter()
    mcq_loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    itc_loss_meter = AverageMeter()
    one_loss_meter = AverageMeter()
    two_loss_meter = AverageMeter()
    three_loss_meter = AverageMeter()
    four_loss_meter = AverageMeter()
    nir_itc_loss_meter = AverageMeter()
    cp_itc_loss_meter = AverageMeter()
    sk_itc_loss_meter = AverageMeter()
    txt_itc_loss_meter = AverageMeter()
    nir_sdm_loss_meter = AverageMeter()
    cp_sdm_loss_meter = AverageMeter()
    sk_sdm_loss_meter = AverageMeter()
    txt_sdm_loss_meter = AverageMeter()
    cone_mmitc_loss_meter = AverageMeter()
    ctwo_mmitc_loss_meter = AverageMeter()
    cthree_mmitc_loss_meter = AverageMeter()
    cfour_mmitc_loss_meter = AverageMeter()
    cone_mmsdm_loss_meter = AverageMeter()
    ctwo_mmsdm_loss_meter = AverageMeter()
    cthree_mmsdm_loss_meter = AverageMeter()
    cfour_mmsdm_loss_meter = AverageMeter()
    vib_kl_loss_meter = AverageMeter()
    single_text_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    best_map = 0.0
    # best_itop1 = 0.0
    # best_ftop1 = 0.0

    # global_step=0

    # top1, map1 = evaluator.eval(model.eval(), use_weight=args.use_weight)
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        if epoch < getattr(args, 'fusion_start_epoch', 25):
            args.use_weight = False
            args.mm_itc_loss_weight = 0.0
            args.single_itc_loss_weight = 1.0
            for name, p in model.named_parameters():
                if 'adapter' in name:
                    p.requires_grad = False
        else:
            args.use_weight = True
            args.mm_itc_loss_weight = 1.0
            args.single_itc_loss_weight = 0.0
            for name, p in model.named_parameters():
                if 'lora' in name or 'tokenizer' in name or 'ln' in name:
                    p.requires_grad = False
                if 'adapter' in name:
                    p.requires_grad = True

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # for name, p in model.named_parameters():
        #     if 'tokenizer' in name or 'vision_encoder' in name:
        #         p.requires_grad = False

        # for name, p in model.named_parameters():
        #     if 'tokenizer' in name or 'lora' in name or 'adapter' in name:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        id_loss_meter.reset()
        ctwo_mca_loss_meter.reset()
        cthree_mca_loss_meter.reset()
        cfour_mca_loss_meter.reset()
        mcm_loss_meter.reset()
        mlm_loss_meter.reset()
        mlm_acc_meter.reset()
        text_itc_loss_meter.reset()
        mcq_loss_meter.reset()
        itc_loss_meter.reset()
        single_text_loss_meter.reset()
        one_loss_meter.reset()
        two_loss_meter.reset()
        three_loss_meter.reset()
        four_loss_meter.reset()
        nir_itc_loss_meter.reset()
        cp_itc_loss_meter.reset()
        sk_itc_loss_meter.reset()
        txt_itc_loss_meter.reset()
        nir_sdm_loss_meter.reset()
        cp_sdm_loss_meter.reset()
        sk_sdm_loss_meter.reset()
        txt_sdm_loss_meter.reset()
        cone_mmsdm_loss_meter.reset()
        ctwo_mmsdm_loss_meter.reset()
        cthree_mmsdm_loss_meter.reset()
        cfour_mmsdm_loss_meter.reset()
        cone_mmitc_loss_meter.reset()
        ctwo_mmitc_loss_meter.reset()
        cthree_mmitc_loss_meter.reset()
        cfour_mmitc_loss_meter.reset()
        vib_kl_loss_meter.reset()
        model.train()

        # evaluator.eval(model.eval())
        # top1, map1 = evaluator.eval(model.eval(), use_weight=args.use_weight)
        
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # batch中的key[images]换成rgbs，书序不要变啊
            batch = {
                ("rgbs" if k == "images" else k): v
                for k, v in batch.items()
            }

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            loss_meter.update(total_loss.item(), batch['rgbs'].shape[0])
            acc_meter.update(ret.get('acc', 0), 1)

            id_loss_meter.update(ret.get('id_loss', 0), batch['rgbs'].shape[0])
            ctwo_mca_loss_meter.update(ret.get('ctwo_mca_loss', 0), batch['rgbs'].shape[0])
            cthree_mca_loss_meter.update(ret.get('cthree_mca_loss', 0), batch['rgbs'].shape[0])
            cfour_mca_loss_meter.update(ret.get('cfour_mca_loss', 0), batch['rgbs'].shape[0])
            mcm_loss_meter.update(ret.get('mcm_loss', 0), batch['rgbs'].shape[0])
            mlm_loss_meter.update(ret.get('mlm_loss', 0), batch['rgbs'].shape[0])
            mlm_acc_meter.update(ret.get('mlm_acc', 0), batch['rgbs'].shape[0])
            text_itc_loss_meter.update(ret.get('text_itc_loss', 0), batch['rgbs'].shape[0])
            mcq_loss_meter.update(ret.get('mcq_loss', 0), batch['rgbs'].shape[0])
            itc_loss_meter.update(ret.get('itc_loss', 0), batch['rgbs'].shape[0])
            nir_itc_loss_meter.update(ret.get('nir_itc_loss', 0), batch['rgbs'].shape[0])
            cp_itc_loss_meter.update(ret.get('cp_itc_loss', 0), batch['rgbs'].shape[0])
            sk_itc_loss_meter.update(ret.get('sk_itc_loss', 0), batch['rgbs'].shape[0])
            txt_itc_loss_meter.update(ret.get('txt_itc_loss', 0), batch['rgbs'].shape[0])
            nir_sdm_loss_meter.update(ret.get('nir_sdm_loss', 0), batch['rgbs'].shape[0])
            cp_sdm_loss_meter.update(ret.get('cp_sdm_loss', 0), batch['rgbs'].shape[0])
            sk_sdm_loss_meter.update(ret.get('sk_sdm_loss', 0), batch['rgbs'].shape[0])
            txt_sdm_loss_meter.update(ret.get('txt_sdm_loss', 0), batch['rgbs'].shape[0])
            cone_mmitc_loss_meter.update(ret.get('cone_mmitc_loss', 0), batch['rgbs'].shape[0])
            ctwo_mmitc_loss_meter.update(ret.get('ctwo_mmitc_loss', 0), batch['rgbs'].shape[0])
            cthree_mmitc_loss_meter.update(ret.get('cthree_mmitc_loss', 0), batch['rgbs'].shape[0])
            cfour_mmitc_loss_meter.update(ret.get('cfour_mmitc_loss', 0), batch['rgbs'].shape[0])
            cone_mmsdm_loss_meter.update(ret.get('cone_mmsdm_loss', 0), batch['rgbs'].shape[0])
            ctwo_mmsdm_loss_meter.update(ret.get('ctwo_mmsdm_loss', 0), batch['rgbs'].shape[0])
            cthree_mmsdm_loss_meter.update(ret.get('cthree_mmsdm_loss', 0), batch['rgbs'].shape[0])
            cfour_mmsdm_loss_meter.update(ret.get('cfour_mmsdm_loss', 0), batch['rgbs'].shape[0])
            vib_kl_loss_meter.update(ret.get('vib_kl_loss', 0), batch['rgbs'].shape[0])
            single_text_loss_meter.update(ret.get('single_text_loss', 0), batch['rgbs'].shape[0])
            one_loss_meter.update(ret.get('one_avg_loss', 0), batch['rgbs'].shape[0])
            two_loss_meter.update(ret.get('two_avg_loss', 0), batch['rgbs'].shape[0])
            three_loss_meter.update(ret.get('three_avg_loss', 0), batch['rgbs'].shape[0])
            four_loss_meter.update(ret.get('four_avg_loss', 0), batch['rgbs'].shape[0])

            optimizer.zero_grad()
            total_loss.backward()
            
            # global_step+=1
            # rank_info = rankallocator.update_and_mask(global_step)
            
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                # logger.info(
                #     f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, itc_loss: {itc_loss_meter.avg:.4f}, single_text_loss: {single_text_loss_meter.avg:.4f}, one_loss: {one_loss_meter.avg:.4f}, two_loss: {two_loss_meter.avg:.4f}, three_loss: {three_loss_meter.avg:.4f}, four_loss: {four_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                # )
                print(
                    f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, ctwo_mmitc_loss: {ctwo_mmitc_loss_meter.avg:.4f}, cthree_mmitc_loss: {cthree_mmitc_loss_meter.avg:.4f}, cfour_mmitc_loss: {cfour_mmitc_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                    # f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, cone_mmsdm_loss: {cone_mmsdm_loss_meter.avg:.4f}, ctwo_mmsdm_loss: {ctwo_mmsdm_loss_meter.avg:.4f}, cthree_mmsdm_loss: {cthree_mmsdm_loss_meter.avg:.4f}, cfour_mmsdm_loss: {cfour_mmsdm_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                    # f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, nir_sdm_loss: {nir_sdm_loss_meter.avg:.4f}, cp_sdm_loss: {cp_sdm_loss_meter.avg:.4f}, sk_sdm_loss: {sk_sdm_loss_meter.avg:.4f}, txt_sdm_loss: {txt_sdm_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                    # f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, nir_itc_loss: {nir_itc_loss_meter.avg:.4f}, cp_itc_loss: {cp_itc_loss_meter.avg:.4f}, sk_itc_loss: {sk_itc_loss_meter.avg:.4f}, txt_itc_loss: {txt_itc_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Lr: {scheduler.get_lr()[0]:.2e}"
                    ,f" nir_itc_loss: {nir_itc_loss_meter.avg:.4f}, cp_itc_loss: {cp_itc_loss_meter.avg:.4f}, sk_itc_loss: {sk_itc_loss_meter.avg:.4f}, txt_itc_loss: {txt_itc_loss_meter.avg:.4f}"
                    ,f" ctwo_mca_loss: {ctwo_mca_loss_meter.avg:.4f}, cthree_mca_loss: {cthree_mca_loss_meter.avg:.4f}, cfour_mca_loss: {cfour_mca_loss_meter.avg:.4f}"
                )
                # print(f"Rank Info: {rank_info}")

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        tb_writer.add_scalar('loss', loss_meter.avg, epoch)
        tb_writer.add_scalar('mcm_loss', mcm_loss_meter.avg, epoch)
        tb_writer.add_scalar('id_loss', id_loss_meter.avg, epoch)
        tb_writer.add_scalar('mlm_loss', mlm_loss_meter.avg, epoch)
        tb_writer.add_scalar('mlm_acc', mlm_acc_meter.avg, epoch)
        tb_writer.add_scalar('text_itc_loss', text_itc_loss_meter.avg, epoch)
        tb_writer.add_scalar('mcq_loss', mcq_loss_meter.avg, epoch)
        tb_writer.add_scalar('itc_loss', itc_loss_meter.avg, epoch)  
        tb_writer.add_scalar('nir_itc_loss', nir_itc_loss_meter.avg, epoch)  
        tb_writer.add_scalar('cp_itc_loss', cp_itc_loss_meter.avg, epoch)  
        tb_writer.add_scalar('sk_itc_loss', sk_itc_loss_meter.avg, epoch)  
        tb_writer.add_scalar('txt_itc_loss', txt_itc_loss_meter.avg, epoch)  
        tb_writer.add_scalar('single_text_loss', single_text_loss_meter.avg, epoch)
        tb_writer.add_scalar('one_loss', one_loss_meter.avg, epoch)
        tb_writer.add_scalar('two_loss', two_loss_meter.avg, epoch)
        tb_writer.add_scalar('three_loss', three_loss_meter.avg, epoch)
        tb_writer.add_scalar('four_loss', four_loss_meter.avg, epoch)
        tb_writer.add_scalar('ctwo_mca_loss', ctwo_mca_loss_meter.avg, epoch)
        tb_writer.add_scalar('cthree_mca_loss', cthree_mca_loss_meter.avg, epoch)
        tb_writer.add_scalar('cfour_mca_loss', cfour_mca_loss_meter.avg, epoch)
        tb_writer.add_scalar('acc', acc_meter.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            # if epoch > 22:
            #     args.use_weight = True
            args.training = False
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1, map1 = evaluator.eval(model.module.eval(), use_weight=args.use_weight)
                else:
                    top1, map1 = evaluator.eval(model.eval(), use_weight=args.use_weight)

                # ftop1 = (ttop1 + stop1 + itop1)/3.0

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("R1_best", **arguments)

                if best_map < map1:
                    best_map = map1
                    arguments2["epoch"] = epoch
                    checkpointer.save("mAP_best", **arguments)

                # if best_itop1 < itop1:
                #     best_itop1 = itop1
                #     arguments3["epoch"] = epoch
                #     checkpointer.save("fusion_best", **arguments)

                # if best_ftop1 < ftop1:
                #     best_ftop1 = ftop1
                #     arguments4["epoch"] = epoch
                #     checkpointer.save("average_best", **arguments)

    print(f"best R1: {best_top1} at epoch {arguments['epoch']}")
    print(f"best mAP: {best_map} at epoch {arguments2['epoch']}")
    # logger.info(f"fusion best R1: {best_itop1} at epoch {arguments3['epoch']}")
    # logger.info(f"average best R1: {best_ftop1} at epoch {arguments4['epoch']}")
