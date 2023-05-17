import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import random

import torch
import math
import numpy as np
# import timm
# import torchvision
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
from utils.utils import build_dataloader, train_one_epoch, validation
from utils.parser import parser_args
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
def train_ddp(rank, world_size, args_dict):

    port = args_dict['port']
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:" + str(port),
        world_size=world_size,
        rank=rank,
    )

    train(args_dict, ddp_gpu=rank)
    cleanup()


# train function
def train(args_dict, ddp_gpu=-1):
    cudnn.benchmark = True

    # set gpu of each multiprocessing
    torch.cuda.set_device(ddp_gpu)

    # define model
    model = DA_model(args_dict.n_classes, load_source_model=False)

    # setting Distributed 
    if args_dict['use_ddp']:   
        # model = DDP(model.to(ddp_gpu))
        model = DDP(model(args_dict).to(ddp_gpu))
    
    # setting optim
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args_dict['lr'], momentum=args_dict['momentum'], weight_decay=args_dict['weight_decay'])
    
    # setting lr scheduler as cosine annealing
    lf = lambda x: ((1 + math.cos(x * math.pi / args_dict['cosanneal_cycle'])) / 2) * (1 - args_dict['lrf']) + args_dict['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)
    num_epochs = 50

    best_epoch = 0
    data_path = args.data_path
    batch_size = 32
    num_workers = 6

    # build dataloader
    source_loader, source_val_loader, target_loader, val_loader = build_dataloader(data_path, batch_size, num_workers)
    best_map = map_50 = validation(model, val_loader)


    # check if folder exist and start summarywriter on main worker
    if is_main_worker(ddp_gpu):
        print("Start Training")
        if not os.path.exists(args_dict['model_save_path']):
            os.mkdir(args_dict['model_save_path'])
        tb_writer = SummaryWriter(args_dict['model_save_path'])
    

    # start training
    for epoch in range(num_epochs):
        loss = train_one_epoch(
            model, 
            optimizer, 
            source_loader, 
            target_loader, 
            epoch
        )
        scheduler.step()

        map_50, map_75, map_small, map_medium, map_large = validation(model, val_loader)
        if map_50 > best_map:
            best_map = map_50
            best_epoch = epoch
            torch.save(model.state_dict(), ...)

        # write info into summarywriter in main worker
        if is_main_worker(ddp_gpu):
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]
            tb_writer.add_scalar(tags[0], loss, epoch)
            tb_writer.add_scalar(tags[1], map_50, epoch)
            tb_writer.add_scalar(tags[2], map_75, epoch)
            tb_writer.add_scalar(tags[3], map_small, epoch)
            tb_writer.add_scalar(tags[4], map_medium, epoch)
            tb_writer.add_scalar(tags[5], map_large, epoch)
            tb_writer.add_scalar(tags[6], optimizer.param_groups[0]['lr'], epoch)

            # save model every two epoch 
            if (epoch % args_dict['save_frequency'] == 0 and epoch >= 10):
                save_path = args_dict['model_save_path'] + "/model_{}_{:.3f}_.pth".format(epoch, map_50)
                torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # get args
    args = parser_args()
    args_dict = vars(args)

    init(args_dict['seed'])

    # train in ddp or not
    if args_dict['use_ddp']:
        n_gpus_per_node = torch.cuda.device_count()
        world_size = n_gpus_per_node
        mp.spawn(train_ddp, nprocs=n_gpus_per_node, args=(world_size, args_dict))
    else:
        train(args_dict)


    
