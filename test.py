from pycocotools.coco import COCO

json_file = 'val.coco.json'
coco = COCO(annotation_file=json_file)
categories = coco.cats
n_classes = len(categories.keys())
print(n_classes)
# ids = list(sorted(coco.imgs.keys()))
# print(len(ids))
# ann_ids = coco.getAnnIds(imgIds=0)
# targets = coco.loadAnns(ann_ids)
# path = coco.loadImgs(0)[0]['file_name']
# # print(ann_ids)
# print(coco.loadCats(1))
# print(targets['bbox'])
# print(path)
# coco_class = dict([(v["id"], v["name"]) for k, v in coco.])

# import os
# a=255
# print(a.div(255))