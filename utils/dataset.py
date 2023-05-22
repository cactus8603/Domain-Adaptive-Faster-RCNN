# import torchvision.transforms as T
import albumentations as A
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import torch
import copy
import cv2
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from glob import glob


def get_transform(mode, *args, **kwargs):

    # only do augmentations while training
    if mode == 'train':
        transform = A.Compose([
            A.Resize(480,480),
            A.RandomBrightnessContrast(p=0.1),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ], p=1.0),
            A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco'))
    elif mode == 'val' :
        transform = A.Compose([
            A.Resize(480,480),
            A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco'))
    
    return transform

class SourceDataset(Dataset):

    def __init__(self, root: str, split: str = "train", transform=None, *args, **kwargs) -> None:
        super().__init__()
        
        # read anno json file
        self.coco = COCO(annotation_file=os.path.join(root, split+'.coco.json'))
        
        # get ids
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
        self.root = root
        self.split = split
        self.transform = transform
        # TODO

    def _load_image(self, index: int):
        path = self.coco.loadImgs(index)[0]['file_name']
        
        # root: path to folder
        # path: org/val/2575.png
        # img = cv2.imread(os.path.join(self.img_path, path))
        img = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))
        # img = np.array(Image.open(os.path.join(self.root, path)))
        return img

    def _load_target(self, index: int):
        target = self.coco.loadAnns(self.coco.getAnnIds(index))
        return target

    def __getitem__(self, index: int):
        
        image = self._load_image(index)
        target = copy.deepcopy(self._load_target(index))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']
        # xmin, ymin, w, h -> xmin, ymin, xmax, ymax
        new_boxes = []
        for box in boxes:
            xmin =  box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        boxes = boxes.reshape(-1,4)
        
        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"]  for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)
        
        # TODO: make sure your image is scaled properly
        # return image and target
        # print(image)
        return image, targ

    def __len__(self) -> int:
        return len(self.ids)
        # return the length of dataset


class TargetDataset(Dataset):

    def __init__(self, root, split, transform=None, *args, **kwargs) -> None:
        super().__init__()
        if split == "fog/val":
            self.coco = COCO(os.path.join(root, split+'.coco.json')) 
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
        else:
            self.ids = list(sorted(os.listdir(os.path.join(root,split))))

        self.root = root
        self.split = split
        self.transform = transform


    def _load_image(self, index: int):
        if self.split == "fog/val":
            path = self.coco.loadImgs(index)[0]['file_name']
            image = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))
            # image = np.array(Image.open(os.path.join(self.root, path)))
        else:
            image = np.array(Image.open(os.path.join(self.root, self.split, self.ids[index])).convert('RGB'))
            # image = np.array(Image.open(os.path.join(self.root, self.split, self.ids[index])))
        return image

    def _load_target(self, index: int):
        target = self.coco.loadAnns(self.coco.getAnnIds(index))
        return target


    def __getitem__(self, index: int):
        # fog/train do not have bbox
        if self.split == "fog/val":
            image = self._load_image(index)
            target = copy.deepcopy(self._load_target(index))
            
            boxes = [t['bbox'] + [t['category_id']] for t in target]
            if self.transform is not None:
                transformed = self.transform(image=image, bboxes=boxes)
                image = transformed['image']
                boxes = transformed['bboxes']
                
            # xmin, ymin, w, h -> xmin, ymin, xmax, ymax
            new_boxes = []
            for box in boxes:
                xmin = box[0]
                ymin = box[1]
                xmax = xmin + box[2]
                ymax = ymin + box[3]
                new_boxes.append([xmin, ymin, xmax, ymax])
            
            boxes = torch.tensor(new_boxes, dtype=torch.float32)
            boxes = boxes.reshape(-1,4)
            targ = {}

            targ["boxes"] = boxes
            targ["labels"] = torch.tensor([t["category_id"]-1  for t in target], dtype=torch.int64)
            targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
            targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)
            
            # TODO: make sure your image is scaled properly
            # return image and target
        
            return image, targ
        # for fog/train
        else:
            image = self._load_image(index)
            # img = self.transform(image)
            if self.transform is not None:
                transformed = self.transform(image=image, bboxes=[])
                
            image = transformed['image']
            boxes = transformed['bboxes']
                
            return image

    def __len__(self) -> int:
        return len(self.ids)
