import json
import logging
from argparse import ArgumentParser
from os import path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image
from fiftyone.utils.voc import VOCBoundingBox

from ..constants import LABEL_MAP

log = logging.getLogger(__name__)


def df_to_coco(df: pd.DataFrame, labelmap: Dict[str, int],
               predictions: bool = False) -> dict:
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    img_id = 0

    for sample in df.itertuples():
        # Read annotation xml

        if not hasattr(sample, 'height') or np.isnan(sample.height) or np.isnan(sample.width):
            width, height = Image.open(sample.filepath).size
        else:
            width, height = sample.width, sample.height

        img_id += 1
        image_info = {
            'file_name': path.split(sample.filepath)[-1],
            'height': height,
            'width': width,
            'id': img_id,
            'eventId': sample.eventId
        }
        output_json_dict['images'].append(image_info)

        if not predictions:
            bbox = [sample.x0, sample.y0,
                    sample.x1 - sample.x0, sample.y1 - sample.y0]

            ann = {
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'bbox': bbox,
                'category_id': labelmap[sample.cls],
                'ignore': 0,
                'segmentation': []  # This script is not for segmentation
            }

            if hasattr(sample, 'detection_score'):
                ann['score'] = sample.detection_score

            ann.update({'image_id': img_id, 'id': bnd_id})
            bnd_id = bnd_id + 1

            output_json_dict['annotations'].append(ann)
        else:
            for i in range(sample.num_detections):
                bbox = VOCBoundingBox(sample.detected_x0[i],
                                      sample.detected_y0[i],
                                      sample.detected_x1[i],
                                      sample.detected_y1[i])

                ann = {
                    'area': (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin),
                    'iscrowd': 0,
                    'image_id': sample.eventId,
                    'id': bnd_id,
                    'bbox': bbox.to_detection_format((width, height)),
                    'score': sample.detection_score[i],
                    'category_id': labelmap[sample.detected_class[i]],
                    'ignore': 0,
                    'segmentation': [],  # This script is not for segmentation
                }

                bnd_id = bnd_id + 1
                output_json_dict['annotations'].append(ann)

    for label, label_id in labelmap.items():
        category_info = {'supercategory': 'none', 'id': label_id,
                         'name': label}

        output_json_dict['categories'].append(category_info)

    return output_json_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('csv_path',
                        help='Path to dataset csv.')
    parser.add_argument('-o', '--output', default='output.json',
                        help='JSON file to write COCO formatted dataset to.')

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    coco_dict = df_to_coco(df, LABEL_MAP)

    with open(args.output, 'w') as f:
        json.dump(coco_dict, f)
