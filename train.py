import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import random

import torch
import math
import numpy as np
import timm
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import logging
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.handlers import create_lr_scheduler_with_warmup
from torch import nn, optim
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tensorboardX import SummaryWriter
from torch.cuda import amp
from utils.dataset import ImgDataSet
from utils.utils import read_spilt_data, get_loader, train_one_epoch, evaluate
from utils.parser import parser_args
from model import DA_model
from pycocotools.coco import COCO


def train_one_epoch(model, optimizer, source_loader, target_loader, epoch):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    pbar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
    for i, ((source_images, source_labels), (target_images)) in enumerate(pbar):

        source_images = list(image.to('cuda', non_blocking=True) for image in source_images) # list of [C, H, W]
        source_labels = [{k: v.to('cuda', non_blocking=True) for k, v in t.items()} for t in source_labels]
        target_images = list(image.to('cuda', non_blocking=True) for image in target_images) # list of [C, H, W]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = model(source_images, source_labels, target_images)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix({'loss': losses.item()})


@torch.no_grad()
def validation(model, data_loader):
    metric = MeanAveragePrecision()
    model.eval()

    for images, targets in tqdm(data_loader):
        images = list(image.to('cuda', non_blocking=True) for image in images)
        predictions = model(images)
        predictions = ...  # postprocess: modify format to meet metric's requirements
        metric.update(predictions, targets)
    
    result = metric.compute()
    return result['map_50']


def build_dataloader():

    def collate_fn(batch):
        ...
        # TODO
        # depends on your code
        # return tuple(zip(*batch))

    # TODO
    from utils.dataset import SourceDataset, TargetDataset, get_transform
    source_dataset = SourceDataset()
    target_dataset = TargetDataset()
    val_dataset = ...

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)

    return source_loader, target_loader, val_loader


model = DA_model(n_classes, load_source_model=False)
model = model.to('cuda')


# TODO
optimizer = ...
scheduler = ...
num_epochs = 50

best_epoch = 0
source_loader, target_loader, val_loader = build_dataloader()
best_map = map_50 = validation(model, val_loader)
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, source_loader, target_loader, epoch)
    scheduler.step()

    map_50 = validation(model, val_loader)
    if map_50 > best_map:
        best_map = map_50
        best_epoch = epoch
        torch.save(model.state_dict(), ...)

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
