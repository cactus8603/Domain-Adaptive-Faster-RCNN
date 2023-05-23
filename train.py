import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random

import torch
import math
import numpy as np
# import timm
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import logging
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
# from ignite.handlers import create_lr_scheduler_with_warmup
# from torch import nn, optim
# from tqdm import tqdm

from tensorboardX import SummaryWriter
# from torch.cuda import amp
from utils.utils import build_dataloader, train_one_epoch, validation, train_source_one_peoch
from utils.parser import create_parser
from model import DA_model
# from pycocotools.coco import COCO

# set seed
def init(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def cleanup():
    dist.destroy_process_group()

def is_main_worker(gpu):
    return (gpu <= 0)

# mp.spawn will pass the value "gpu" as rank
def train_ddp(rank, world_size, args):
    print("start dist")
    port = args.port
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:" + str(port),
        world_size=world_size,
        rank=rank,
    )

    train(args, ddp_gpu=rank)
    cleanup()

# train function
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("ddp_gpu:", ddp_gpu)
    # cudnn.benchmark = True

    # set gpu of each multiprocessing
    # torch.cuda.set_device(ddp_gpu)

    # define model
    model = DA_model(args.n_classes, device, load_source_model=True)

    # source model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
    # in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    # model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, args.n_classes)


    model = model.to(device)

    
    # setting optim
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # setting lr scheduler as cosine annealing
    lf = lambda x: ((1 + math.cos(x * math.pi / args.cosanneal_cycle)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)
    num_epochs = args.epoch

    best_epoch = 0
    data_path = args.data_path
    batch_size = args.batch_size
    num_workers = args.num_workers

    # build dataloader
    source_loader, source_val_loader, target_loader, val_loader = build_dataloader(data_path, batch_size, num_workers)
    # best_map = map_50 = validation(model, val_loader)
    best_map = 0
    map_50 = 0
    best_epoch = 0

    # check if folder exist and start summarywriter on main worker
    print("Start Training")
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    tb_writer = SummaryWriter(args.save_path)
    

    # start training
    for epoch in range(num_epochs):

        ### for source 
        # loss = train_source_one_peoch(
        #     model, 
        #     optimizer, 
        #     source_loader, 
        #     epoch, 
        #     device=device
        # )

        # scheduler.step()

        # map_50, map_75, map_small, map_medium, map_large = validation(model, source_val_loader)

        # print(device)
        ### for domain
        loss = train_one_epoch(
            model, 
            optimizer, 
            source_loader, 
            target_loader, 
            args
            # accumulation=args.accumulation
        )

        scheduler.step()

        map_50, map_75, map_small, map_medium, map_large = validation(model, val_loader)

        # write info into summarywriter
        tags = ["loss", "map_50", "map_75", "map_small", "map_medium", "map_large", "lr"]
        tb_writer.add_scalar(tags[0], loss, epoch)
        tb_writer.add_scalar(tags[1], map_50, epoch)
        tb_writer.add_scalar(tags[2], map_75, epoch)
        tb_writer.add_scalar(tags[3], map_small, epoch)
        tb_writer.add_scalar(tags[4], map_medium, epoch)
        tb_writer.add_scalar(tags[5], map_large, epoch)
        tb_writer.add_scalar(tags[6], optimizer.param_groups[0]['lr'], epoch)

        # save model of 0%, 33%, 66%, 100%
        if (epoch == 0):
            save_path = os.path.join(args.save_path, "model_0.pth")
            torch.save(model.state_dict(), save_path)
        elif (epoch == int(args.epoch * 0.33 - 1)):
            save_path = os.path.join(args.save_path, "model_33.pth")
            torch.save(model.state_dict(), save_path)
        elif (epoch == int(args.epoch * 0.66 - 1)):
            save_path = os.path.join(args.save_path, "model_66.pth")
            torch.save(model.state_dict(), save_path)
        elif (epoch == args.epoch - 1):
            save_path = os.path.join(args.save_path, "model_100.pth")
            torch.save(model.state_dict(), save_path)

        # save model of best epoch
        if map_50 > best_map:
            best_map = map_50
            best_epoch = epoch
            save_path = os.path.join(args.save_path, "best_model.pth")
            torch.save(model.state_dict(), save_path)

      

if __name__ == '__main__':
    # get args
    args = create_parser()
    # args_dict = vars(args)
    # print(args)
    init(args.seed)

    # train in ddp or not
    train(args)

    # if args.use_ddp:
    #     n_gpus_per_node = torch.cuda.device_count()
    #     world_size = n_gpus_per_node
    #     mp.spawn(train_ddp, nprocs=n_gpus_per_node, args=(world_size, args))
    # else:
        # train(args_dict)


    
