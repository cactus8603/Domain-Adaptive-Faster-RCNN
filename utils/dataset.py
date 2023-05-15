import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import torch
import copy
import cv2
from PIL import Image
from glob import glob


def get_transform(*args, **kwargs):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224,224),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    
    # TODO
    # perform data augmentation
    # convert data type and value range
    # return transform


class SourceDataset(Dataset):

    def __init__(self, img_path, json_path, transform=None, *args, **kwargs) -> None:
        super().__init__()

        self.coco = COCO(annotation_file=json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_path = img_path
        # TODO

    def _load_image(self, index: int):
        path = self.coco.loadImgs(index)[0]['file_name']
        
        # img_path: path to folder
        # ex path:org/val/2575.png
        img = cv2.imread(os.path.join(self.img_path, path))
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        # return image

    def _load_target(self, index: int):
        target = self.coco.loadAnns(self.coco.getAnnIds(index))
        ...
        # return target

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
        
        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"]  for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)
        
        # TODO: make sure your image is scaled properly
        # return image and target

    def __len__(self) -> int:
        return len(self.ids)
        # return the length of dataset


class TargetDataset(Dataset):

    def __init__(self, root: str, split: str = "train", transform=None, *args, **kwargs) -> None:
        super().__init__()
        # TODO

    def _load_image(self, index: int):
        ...
        # return image

    def __getitem__(self, index: int):
        ...
        # return image and target

    def __len__(self) -> int:
        ...
        # return the length of dataset

class ValDataset(Dataset):

    def __init__(self, )
