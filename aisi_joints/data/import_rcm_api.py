"""
Preprocesses .csv files from the RCM API by merging .csv fields with labels
and bounding boxes, and filtering based on presence of class label and
available image data. The matching is done using eventId UUID.

The script outputs a .csv file containing all data samples that have
class label and corresponding image, including absolute path to file.
The output csv can then be used to produce tfrecord files for TFOD.

Run this script as
```
python -m aisi_joints.data.import_rcm_api --labels /path/to/.csv
    --boxes /path/to/other/csv --images /path/to/image/dir
"""
import logging
import os.path as path
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..constants import CLASS_DEFECT, CLASS_OK, LABEL_MAP
from .common import find_images, find_labels, write_pbtxt

log = logging.getLogger(__name__)


def import_rcm_api(labels_pth: List[str], boxes_pth: List[str],
                   images_pth: List[str],
                   deviations_only: bool = False) \
        -> Tuple[pd.DataFrame, dict]:
    labels_df = find_labels(labels_pth)

    orig_len = len(labels_df)

    # filter out samples that have no class labels
    labels_df = labels_df[labels_df['deviationId'].notna()
                          | labels_df['ratingStatus'].notna()]

    log.info(f'Registered {len(labels_df)} labels from {len(labels_pth)} '
             f'files, dropped {orig_len - len(labels_df)} due to missing class '
             f'labels.')

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
    df = pd.merge(labels_df, boxes_df[['eventId', 'platformID', 'sessionId',
                                       'x0', 'x1', 'y0', 'y1']],
                  left_on='eventId', right_on='eventId')

    log.info(f'Found {len(df)} samples with matching labels and boxes.')

    # add class column based on deviationID and ratingStatus values
    df['cls'] = np.nan
    df.loc[df['deviationId'].notna(), 'cls'] = CLASS_DEFECT

    if deviations_only:
        df = df.drop(df.loc[(df['ratingStatus'] == 'IND_TRUE_POSITIVE') & ~df['deviationId'].notna()].index)
    else:
        df.loc[df['ratingStatus'] == 'IND_TRUE_POSITIVE', 'cls'] = CLASS_DEFECT

    df.loc[df['ratingStatus'] == 'IND_FALSE_POSITIVE', 'cls'] = CLASS_OK

    images_df = find_images(images_pth)

    # merge and filter based on eventId
    df = df.merge(images_df, left_on='eventId', right_on='eventId')

    log.info(f'Total number of labeled samples: {len(df)}.')
    log.info(f'Total number of non-defects: '
             f'{len(df[df["cls"] == CLASS_OK])}.')
    log.info(f'Total number of defects: '
             f'{len(df[df["cls"] == CLASS_DEFECT])}.')

    return df, LABEL_MAP


def main(args: Namespace):
    df, label_map = import_rcm_api(args.labels, args.boxes, args.images)
    if args.output is not None:
        log.info(f'Writing output .csv to {path.abspath(args.output)}.')
        df.to_csv(args.output, index=False)

        basename = path.splitext(args.output)[0]
        write_pbtxt(label_map, basename + '_labelmap.pbtxt')


if __name__ == '__main__':
    from ..utils.logging import setup_logger

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', help='Path to images.')
    parser.add_argument('-b', '--boxes', nargs='+',
                        help='Path to.csv files with bounding boxes.')
    parser.add_argument('-l', '--labels', nargs='+',
                        help='Path to .csv files with data labels and other '
                             'metadata.')
    parser.add_argument('-o', '--output', default='output.csv',
                        help='Output csv name')

    args = parser.parse_args()

    setup_logger()
    main(args)
