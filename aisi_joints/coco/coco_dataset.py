"""
Create COCO dataset on disk from .csv file
"""
import json
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from os import path

import pandas as pd

from ..constants import LABEL_MAP
from ..utils.logging import setup_logger
from ..eval.coco_format import df_to_coco


log = logging.getLogger(__name__)


def create_coco_dataset(df: pd.DataFrame, output_dir: str):

    splits = {}

    if 'split' in df.columns:
        for split in df['split'].unique():
            partial = df[df['split'] == split]

            annotations = df_to_coco(partial, LABEL_MAP)

            splits[split] = annotations
    else:
        annotations = df_to_coco(df, LABEL_MAP)
        splits['dataset'] = annotations

    if not path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ann_dir = path.join(output_dir, 'annotations')
    if not path.exists(ann_dir):
        os.makedirs(ann_dir, exist_ok=True)

    log.info(f'Writing annotations to {ann_dir}.')
    for split in splits.keys():
        annotations = splits[split]

        with open(path.join(ann_dir, f'{split}.json'), 'w') as f:
            json.dump(annotations, f, indent=4)

    for split in splits.keys():
        split_dir = path.join(output_dir, split)

        if not path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)

        log.info(f'Copying images for split {split} to {split_dir}.')

        if 'split' in df.columns:
            filepaths = df[df['split'] == split].filepath
        else:
            filepaths = df.filepath

        for file in filepaths:
            shutil.copy(file, path.join(split_dir, path.split(file)[-1]))

        log.info('Done.')


def main(args: Namespace):
    df = pd.read_csv(args.csv_path)

    create_coco_dataset(df, args.output_dir)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('csv_path',
                        help='Path to csv dataset to create dataset from.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        default='output',
                        help='Directory to write dataset to.')

    args = parser.parse_args()

    setup_logger()
    main(args)

