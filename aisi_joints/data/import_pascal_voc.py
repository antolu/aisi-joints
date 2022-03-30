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

from .common import find_images, find_labels, write_pbtxt

log = logging.getLogger(__name__)

CLASS_DEFECT = 'DEFECT'
CLASS_OK = 'OK'


def xml_to_df(xml_dir: str) -> pd.DataFrame:
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

        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (eventId,
                     member.find('name').text,
                     float(bndbox.find('xmin').text),
                     float(bndbox.find('xmax').text),
                     float(bndbox.find('ymin').text),
                     float(bndbox.find('ymax').text),
                     )
            xml_list.append(value)

        if len(root.findall('object')) == 0:
            log.warning(f'No bounding boxes found for {xml_file}.')

    column_name = ['eventId', 'class', 'x0', 'x1', 'y0', 'y1']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df


def import_pascal_voc(labels_pth: List[str], xmls_pth: List[str],
                      images_pth: List[str]) \
        -> Tuple[pd.DataFrame, dict]:
    metadata_df = find_labels(labels_pth)

    # process bounding boxes
    xmls_df = pd.DataFrame()
    for xml_pth in xmls_pth:
        if not path.isdir(xml_pth):
            raise FileNotFoundError(f'Could not find dir at {xml_pth}.')

        xml_df = xml_to_df(xml_pth)

        xmls_df = pd.concat([xmls_df, xml_df])

    # rename class label and reorder columns
    xmls_df['cls'] = np.nan
    xmls_df.loc[xmls_df['class'] == 'OK_Joint', 'cls'] = CLASS_OK
    xmls_df.loc[xmls_df['class'] == 'DEF_Joint', 'cls'] = CLASS_DEFECT
    xmls_df = xmls_df.drop(columns=['class'])
    cols = xmls_df.columns[:-1].insert(1, 'cls')
    xmls_df = xmls_df[cols]

    log.info(f'Registered {len(xmls_df)} .xmls from {len(xmls_pth)} directories.')

    images_df = find_images(images_pth)

    log.info('Matching labels to images...')
    labels_df = pd.merge(xmls_df, images_df, on='eventId', how='inner')
    log.info(f'Found {len(labels_df)} samples with matching labels and images.')

    # merge labels and boxes df
    log.info('Matching samples to RCM metadata...')
    df = pd.merge(metadata_df, labels_df, on='eventId', how='inner')

    log.info(f'Found {len(df)} samples with RCM metadata.')

    # merge and filter based on eventId
    log.info(f'Total number of labeled samples: {len(df)}.')
    log.info(f'Total number of non-defects: '
             f'{len(df[df["cls"] == CLASS_OK])}.')
    log.info(f'Total number of defects: '
             f'{len(df[df["cls"] == CLASS_DEFECT])}.')

    label_map = {
        CLASS_OK: 0,
        CLASS_DEFECT: 1
    }

    return df, label_map


def main(args: Namespace):
    df, label_map = import_pascal_voc(
        args.rcm_csv, args.xmls, args.images)

    if args.output is not None:
        log.info(f'Writing output .csv to {path.abspath(args.output)}.')
        df.to_csv(args.output, index=False)

        basename = path.splitext(args.output)[0]
        write_pbtxt(label_map, basename + '_labelmap.pbtxt')


if __name__ == '__main__':
    from ..utils.logging import setup_logger

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', help='Path to images.')
    parser.add_argument('-x', '--xmls', nargs='+',
                        help='Path to directory with .xml files with bounding boxes and label.')
    parser.add_argument('-l', '--rcm-csv', dest='rcm_csv', nargs='+',
                        help='Path to .csv files with RCM API metadata.')
    parser.add_argument('-o', '--output', default='output.csv',
                        help='Output csv name')

    args = parser.parse_args()

    setup_logger()
    main(args)
