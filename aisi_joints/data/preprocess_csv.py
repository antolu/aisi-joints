"""
Preprocesses .csv files from the RCM API by merging .csv fields with labels
and bounding boxes, and filtering based on presence of class label and
available image data. The matching is done using eventId UUID.

The script outputs a .csv file containing all data samples that have
class label and corresponding image, including absolute path to file.
The output csv can then be used to produce tfrecord files for TFOD.

Run this script as
```
python -m aisi_joints.data.preprocess_csv --labels /path/to/.csv --boxes /path/to/other/csv --images /path/to/image/dir
"""
from typing import List, Optional

import pandas as pd
import re
import json
import numpy as np
import os.path as path
import os


import logging

log = logging.getLogger(__name__)


def preprocess_csv(labels_pth: List[str], boxes_pth: List[str], images_pth: List[str],
                   output: Optional[str] = None):

    # process label files first
    labels_df = pd.DataFrame()
    for label_pth in labels_pth:
        if not path.isfile(label_pth):
            raise FileNotFoundError(f'Could not find label file at {label_pth}.')

        with open(label_pth) as f:
            label_df = pd.read_csv(f)
        label_df = label_df.drop('rating', axis=1)  # remove json formatted column
        labels_df = pd.concat([labels_df, label_df])

    orig_len = len(labels_df)

    # filter out samples that have no class labels
    labels_df = labels_df[labels_df['deviationId'].notna() | labels_df['ratingStatus'].notna()]

    log.info(f'Registered {len(labels_df)} labels from {len(labels_pth)} files, dropped {orig_len - len(labels_df)} due to missing class labels.')

    # process bounding boxes
    boxes_df = pd.DataFrame()
    for box_pth in boxes_pth:
        if not path.isfile(box_pth):
            raise FileNotFoundError(f'Could not find box file at {box_pth}.')

        with open(box_pth) as f:
            box_df = pd.read_csv(f)
        boxes_df = pd.concat([boxes_df, box_df])

    log.info(f'Registered {len(boxes_df)} boxes from {len(boxes_pth)} files.')

    # merge labels and boxes df
    log.info('Matching labels to boxes...')
    df = pd.merge(labels_df, boxes_df[['eventId', 'platformID', 'sessionId', 'x0', 'x1', 'y0', 'y1']],
                  left_on='eventId', right_on='eventId')

    log.info(f'Found {len(df)} samples with matching labels and boxes.')

    # add class column based on deviationID and ratingStatus values
    df['class'] = np.nan
    df.loc[df['deviationId'].notna(), 'class'] = 1
    df.loc[df['ratingStatus'] == 'IND_TRUE_POSITIVE', 'class'] = 1
    df.loc[df['ratingStatus'] == 'IND_FALSE_POSITIVE', 'class'] = 0

    # iterate over all image files to construct index
    images_idx = list()
    regex = re.compile(r'.*_(?P<uuid>.+)\.(png|jpg)')
    for image_pth in images_pth:
        if not path.isdir(image_pth):
            raise NotADirectoryError(f'{image_pth} is not a directory.')

        for f in os.listdir(image_pth):
            m = regex.match(f)

            if m:
                images_idx.append({'eventId': m.group('uuid'), 'filepath': path.abspath(path.join(image_pth, f))})

    images_df = pd.DataFrame(images_idx)
    log.info(f'Registered {len(images_idx)} images in {len(images_pth)} directories.')

    # merge and filter based on eventId
    df = df.merge(images_df, left_on='eventId', right_on='eventId')

    log.info(f'Total number of labeled samples: {len(df)}.')
    log.info(f'Total number of non-defects: {len(df[df["class"] == 0])}.')
    log.info(f'Total number of defects: {len(df[df["class"] == 1])}.')

    if output is not None:
        log.info(f'Writing output .csv to {path.abspath(output)}.')
        df.to_csv(output)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--images', nargs='+', help='Path to images.')
    parser.add_argument('--boxes', nargs='+', help='Path to.csv files with bounding boxes.')
    parser.add_argument('--labels', nargs='+', help='Path to .csv files with data labels and other metadata.')
    parser.add_argument('--output', help='Output csv name', default='output.csv')

    args = parser.parse_args()

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    log.addHandler(ch)

    preprocess_csv(args.labels, args.boxes, args.images, args.output)
