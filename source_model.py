import torch
from torchvision import models , utils 
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# import pandas as pd
from pycocotools.coco import COCO
import os
from timm.scheduler.cosine_lr import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes

from utils.dataset import SourceDataset, get_transform
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    

def train_one_epoch(model, optimizer, loader, epoch,lr_scheduler):
    # model.to(device)
    model.train()
    
    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1.0 / 1000 # do lr warmup
    #     warmup_iters = min(1000, len(loader) - 1)
        
    #     lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step(loss_value) # 
        
    # all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    # print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
    #     epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
    #     all_losses_dict['loss_classifier'].mean(),
    #     all_losses_dict['loss_box_reg'].mean(),
    #     all_losses_dict['loss_rpn_box_reg'].mean(),
    #     all_losses_dict['loss_objectness'].mean()
    # ))

def validation(model, data_loader):
    metric = MeanAveragePrecision()
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            images = list(image.to(device) for image in images)

            predictions = model(images)
            # predictions = ...  # postprocess: modify format to meet metric's requirements
            metric.update(predictions, targets)

        
        result = metric.compute()
    print(result)
    return result['map_50']

def collate_fn(batch):
    return tuple(zip(*batch))

epochs = 60 
batch  = 8

datasets_path = "./dataset/"

coco = COCO(os.path.join(datasets_path, "org/train.coco.json"))
categories = coco.cats
n_classes = len(categories.keys())
print(n_classes)
train_dataset = SourceDataset(root=datasets_path, split="org/train", transform=get_transform(mode='train') )
train_loader  = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=0, collate_fn=collate_fn)

val_dataset   = SourceDataset(root=datasets_path, split="org/val", transform=get_transform(mode='val'))
val_loader    = DataLoader(val_dataset, batch_size=batch, shuffle=True, num_workers=0, collate_fn=collate_fn)

# model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True, weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8)
lr_scheduler = CosineLRScheduler(
                        optimizer      = optimizer, 
                        t_initial      = epochs // 5,
                        warmup_t       = 20, 
                        warmup_lr_init = 5e-4,
                        cycle_limit    = 5,
                        k_decay        = 0.5,
                        lr_min         = 1e-4
                    )

epoch_list = []
map_list = []

model.to(device)
for epoch in range(epochs):
    print(f"epoch : {epoch} / {epochs} : ")
    train_one_epoch(model, optimizer, train_loader, epoch, lr_scheduler)
    map_50 = validation(model=model, data_loader=val_loader )
    
    if map_50 > 0.32:
        torch.save( model, f"./model/model_{epoch}_{map_50:.4f}.pt" )
    if epoch %5 == 0:
        epoch_list.append(epoch)
        map_list.append(map_50)
        
fig = plt.figure(figsize=(10,5))
plt.plot(epoch_list, map_list, marker ='.', markersize=10, linewidth=2.0) 
plt.xlabel('epoch') 
plt.ylabel('mAP') 
fig.savefig('source_model_map.jpg', bbox_inches='tight', dpi=150)

