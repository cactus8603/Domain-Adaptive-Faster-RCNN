import os
# import random
import torch
# import numpy as np
# import torch.nn.functional as F
from tqdm import tqdm
# from glob import glob
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast as autocast
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .dataset import SourceDataset, TargetDataset, get_transform


def build_dataloader(data_path, batch_size, num_workers):

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    source_dataset = SourceDataset(root=data_path, split="org/train", transform=get_transform(mode='train'))
    source_val_dataset = SourceDataset(root=data_path, split="org/val", transform=get_transform(mode='val'))
    target_dataset = TargetDataset(root=data_path, split="fog/train", transform=get_transform(mode='train'))
    val_dataset = TargetDataset(root=data_path, split="fog/val", transform=get_transform(mode='val') )

    source_loader = DataLoader(source_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    source_val_loader = DataLoader(source_val_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    target_loader = DataLoader(target_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)

    return source_loader, source_val_loader, target_loader, val_loader


def train_one_epoch(model, optimizer, source_loader, target_loader, epoch, device):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    pbar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
    for i, ((source_images, source_labels), (target_images)) in enumerate(pbar):

        source_images = list(image.to(device, non_blocking=True) for image in source_images) # list of [C, H, W]
        source_labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in source_labels]
        target_images = list(image.to(device, non_blocking=True) for image in target_images) # list of [C, H, W]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=False):
            loss_dict = model(source_images, source_labels, target_images)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix({'loss': losses.item()})
    return losses.item()

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
    return result['map_50'], result['map_75'], result['map_small'], result['map_medium'], result['map_large']