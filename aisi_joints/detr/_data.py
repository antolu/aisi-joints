from typing import Dict

import torchvision
import os
import numpy as np
import torch
from pycocotools.coco import COCO

from ..constants import LABEL_MAP


def collate_fn(feature_extractor, batch: dict):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = dict()
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, dataset_folder: str, split_name: str,
                 feature_extractor):
        ann_file = os.path.join(dataset_folder, 'annotations',
                                f'{split_name}.json')
        img_folder = os.path.join(dataset_folder, split_name)

        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target,
                                          return_tensors="pt")
        pixel_values = encoding[
            "pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


def results_to_coco(coco_gt: COCO, results: list,
                  labelmap: Dict[int, str] = None) -> dict:
    if labelmap is None:
        labelmap = LABEL_MAP

    output_json_dict = {
        "images": list(coco_gt.imgs.values()),
        "type": "instances",
        "annotations": [],
        "categories": list(coco_gt.cats.values())
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?

    for res in results:
        for img_id, output in res.items():
            boxes = convert_to_xywh(output['boxes']).tolist()
            scores = output['scores'].tolist()
            labels = output['labels'].tolist()

            for i, bbox in enumerate(boxes):
                ann = {
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'image_id': img_id,
                    'id': bnd_id,
                    'bbox': bbox,
                    'score': scores[i],
                    'category_id': labels[i] + 1,
                    'ignore': 0,
                    'segmentation': [],  # This script is not for segmentation
                }

                bnd_id = bnd_id + 1
                output_json_dict['annotations'].append(ann)

    return output_json_dict


def convert_to_xywh(boxes: torch.tensor):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
