"""
Preprocesses .xml and image files in PASCAL VOC format together with
RCM API .csv files with context information. Bounding boxes and labels
are extracted from the .xml files, and matched to image using eventID.
 merging .csv fields with labels

The script outputs a .csv file containing all data samples that have
class label and corresponding image, including absolute path to file.
The output csv can then be used to produce tfrecord files for TFOD.

Run this script as
```
python -m aisi_joints.data.import_pascal_voc --labels /path/to/.csv
    --boxes /path/to/other/csv --images /path/to/image/dir
"""
import glob
import logging
import os
import os.path as path
import re
import xml.etree.ElementTree as ET
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..constants import CLASS_OK, CLASS_DEFECT, LABEL_MAP
from .common import find_images, find_labels, write_pbtxt

log = logging.getLogger(__name__)


def xml_to_df(xml_dir: str, pad_bndbox: int = 10) -> pd.DataFrame:
    regex = re.compile(r'.*_(?P<uuid>.+)\.(xml)')

    xml_list = []
    for xml_file in os.listdir(xml_dir):
        xml_file = path.join(xml_dir, xml_file)
        m = regex.match(xml_file)
        if not m:
            log.warning(f'eventId not found for file {xml_file}.')
            continue

        eventId = m.group('uuid')

        tree = ET.parse(xml_file)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        for member in root.findall('object'):
            bndbox = member.find('bndbox')

            x0 = int(float(bndbox.find('xmin').text))
            x1 = int(float(bndbox.find('xmax').text))
            y0 = int(float(bndbox.find('ymin').text))
            y1 = int(float(bndbox.find('ymax').text))

            x0 = max(x0 - pad_bndbox, 0)
            x1 = min(x1 + pad_bndbox, width)
            y0 = max(y0 - pad_bndbox, 0)
            y1 = min(y1 + pad_bndbox, height)

            value = (
                eventId,
                member.find('name').text,
                width,
                height,
                x0,
                x1,
                y0,
                y1,
            )
            xml_list.append(value)

        if len(root.findall('object')) == 0:
            log.warning(f'No bounding boxes found for {xml_file}.')

    column_name = [
        'eventId',
        'class',
        'width',
        'height',
        'x0',
        'x1',
        'y0',
        'y1',
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df


def import_pascal_voc(
    labels_pth: List[str],
    xmls_pth: List[str],
    images_pth: List[str],
    pad_bndbox: int = 10,
) -> Tuple[pd.DataFrame, dict]:
    metadata_df = find_labels(labels_pth)

    # process bounding boxes
    xmls_df = pd.DataFrame()
    for xml_pth in xmls_pth:
        if not path.isdir(xml_pth):
            raise FileNotFoundError(f'Could not find dir at {xml_pth}.')

        xml_df = xml_to_df(xml_pth, pad_bndbox)

        xmls_df = pd.concat([xmls_df, xml_df])

    # rename class label and reorder columns
    xmls_df['cls'] = np.nan
    xmls_df.loc[xmls_df['class'] == 'OK_Joint', 'cls'] = CLASS_OK
    xmls_df.loc[xmls_df['class'] == 'DEF_Joint', 'cls'] = CLASS_DEFECT
    xmls_df = xmls_df.drop(columns=['class'])
    cols = xmls_df.columns[:-1].insert(1, 'cls')
    xmls_df = xmls_df[cols]

    log.info(
        f'Registered {len(xmls_df)} .xmls from {len(xmls_pth)} directories.'
    )

    images_df = find_images(images_pth, find_dims=False)

    log.info('Matching labels to images...')
    labels_df = pd.merge(xmls_df, images_df, on='eventId', how='inner')
    log.info(
        f'Found {len(labels_df)} samples with matching labels and images.'
    )

    # merge labels and boxes df
    log.info('Matching samples to RCM metadata...')
    df = pd.merge(metadata_df, labels_df, on='eventId', how='inner')

    log.info(f'Found {len(df)} samples with RCM metadata.')

    # merge and filter based on eventId
    log.info(f'Total number of labeled samples: {len(df)}.')
    log.info(
        f'Total number of non-defects: ' f'{len(df[df["cls"] == CLASS_OK])}.'
    )
    log.info(
        f'Total number of defects: ' f'{len(df[df["cls"] == CLASS_DEFECT])}.'
    )

    return df, LABEL_MAP


def main(args: Namespace):
    df, label_map = import_pascal_voc(args.rcm_csv, args.xmls, args.images)

    if args.output is not None:
        log.info(f'Writing output .csv to {path.abspath(args.output)}.')
        df.to_csv(args.output, index=False)

        basename = path.splitext(args.output)[0]
        write_pbtxt(label_map, basename + '_labelmap.pbtxt')


if __name__ == '__main__':
    from .._utils.logging import setup_logger

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', help='Path to images.')
    parser.add_argument(
        '-x',
        '--xmls',
        nargs='+',
        help='Path to directory with .xml files with bounding boxes and label.',
    )
    parser.add_argument(
        '-l',
        '--rcm-csv',
        dest='rcm_csv',
        nargs='+',
        help='Path to .csv files with RCM API metadata.',
    )
    parser.add_argument(
        '-o', '--output', default='output.csv', help='Output csv name'
    )

    args = parser.parse_args()

    setup_logger()
    main(args)
