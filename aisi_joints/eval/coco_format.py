import json
import logging
from argparse import ArgumentParser
from typing import Dict
from PIL import Image

import pandas as pd

from ..constants import LABEL_MAP

log = logging.getLogger(__name__)


def df_to_coco(df: pd.DataFrame, labelmap: Dict[str, int]):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?

    for sample in df.itertuples():
        # Read annotation xml
        image = Image.open(sample.filepath)
        width, height = image.size

        image_info = {
            'file_name': sample.filepath,
            'height': height,
            'width': width,
            'id': sample.eventId
        }

        output_json_dict['images'].append(image_info)

        o_width = sample.x1 - sample.x0
        o_height = sample.y1 - sample.y0

        ann = {
            'area': o_width * o_height,
            'iscrowd': 0,
            'bbox': [sample.x0, sample.y0, o_width, o_height],
            'category_id': labelmap[sample.cls],
            'ignore': 0,
            'segmentation': []  # This script is not for segmentation
        }

        if hasattr(sample, 'detection_score'):
            ann['score'] = sample.detection_score

        ann.update({'image_id': sample.eventId, 'id': bnd_id})
        bnd_id = bnd_id + 1

        output_json_dict['annotations'].append(ann)

    for label, label_id in labelmap.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}

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
