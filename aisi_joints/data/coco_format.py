"""This script is designed to be a bridge between the aisi_joints dataframe
format and the more common COCO format. This script can be run to convert a
partitioned .csv to an output directory with COCO format ready to be used with
other object detection methods.

For instructions, run the script     python -m
aisi_joints.dat.coco_format --help
"""
import json
import logging
import os
import shutil
from argparse import ArgumentParser
from argparse import Namespace
from os import path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from .._utils.logging import setup_logger
from ..constants import LABEL_MAP

log = logging.getLogger(__name__)


def df_to_coco(
    df: pd.DataFrame, labelmap: Dict[str, int], predictions: bool = False
) -> dict:
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    img_id = 0

    for sample in df.itertuples():
        # Read annotation xml

        if (
            not hasattr(sample, "height")
            or np.isnan(sample.height)
            or np.isnan(sample.width)
        ):
            width, height = Image.open(sample.filepath).size
        else:
            width, height = sample.width, sample.height

        img_id += 1
        image_info = {
            "file_name": path.split(sample.filepath)[-1],
            "height": height,
            "width": width,
            "id": img_id,
            "eventId": sample.eventId,
        }
        output_json_dict["images"].append(image_info)

        if not predictions:
            bbox = [
                sample.x0,
                sample.y0,
                sample.x1 - sample.x0,
                sample.y1 - sample.y0,
            ]

            ann = {
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "bbox": bbox,
                "category_id": labelmap[sample.cls],
                "ignore": 0,
                "segmentation": [],  # This script is not for segmentation
            }

            if hasattr(sample, "detection_score"):
                ann["score"] = sample.detection_score

            ann.update({"image_id": img_id, "id": bnd_id})
            bnd_id = bnd_id + 1

            output_json_dict["annotations"].append(ann)
        elif sample.num_detections > 0:
            x0 = list(map(int, sample.detected_x0.split(";")))
            x1 = list(map(int, sample.detected_x1.split(";")))
            y0 = list(map(int, sample.detected_y0.split(";")))
            y1 = list(map(int, sample.detected_y1.split(";")))
            detection_score = list(map(float, sample.detection_score.split(";")))
            detected_class = sample.detected_class.split(";")
            for i in range(sample.num_detections):
                bbox = [
                    x0[i],
                    y0[i],
                    x1[i] - x0[i],
                    y1[i] - y0[i],
                ]  # x0, y0, w, h

                ann = {
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "image_id": img_id,
                    "id": bnd_id,
                    "bbox": bbox,
                    "score": detection_score[i],
                    "category_id": labelmap[detected_class[i]],
                    "ignore": 0,
                    "segmentation": [],  # This script is not for segmentation
                }

                bnd_id = bnd_id + 1
                output_json_dict["annotations"].append(ann)

    for label, label_id in labelmap.items():
        category_info = {
            "supercategory": "none",
            "id": label_id,
            "name": label,
        }

        output_json_dict["categories"].append(category_info)

    return output_json_dict


def write_coco_dataset(df: pd.DataFrame, output_dir: str):
    splits = {}

    if "split" in df.columns:
        for split in df["split"].unique():
            partial = df[df["split"] == split]

            annotations = df_to_coco(partial, LABEL_MAP)

            splits[split] = annotations
    else:
        annotations = df_to_coco(df, LABEL_MAP)
        splits["dataset"] = annotations

    if not path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ann_dir = path.join(output_dir, "annotations")
    if not path.exists(ann_dir):
        os.makedirs(ann_dir, exist_ok=True)

    log.info(f"Writing annotations to {ann_dir}.")
    for split in splits.keys():
        annotations = splits[split]

        with open(path.join(ann_dir, f"{split}.json"), "w") as f:
            json.dump(annotations, f, indent=4)

    for split in splits.keys():
        split_dir = path.join(output_dir, split)

        if not path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)

        log.info(f"Copying images for split {split} to {split_dir}.")

        if "split" in df.columns:
            filepaths = df[df["split"] == split].filepath
        else:
            filepaths = df.filepath

        for file in filepaths:
            shutil.copy(file, path.join(split_dir, path.split(file)[-1]))

    log.info("Done.")


def main(args: Namespace):
    df = pd.read_csv(args.csv_path)

    write_coco_dataset(df, args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("csv_path", help="Path to csv dataset to create dataset from.")
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default="output",
        help="Directory to write dataset to.",
    )

    args = parser.parse_args()

    setup_logger()
    main(args)
