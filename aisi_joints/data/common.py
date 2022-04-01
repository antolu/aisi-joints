import logging
import os
import re
from os import path
from typing import List, Dict

import pandas as pd

log = logging.getLogger(__name__)


CLASS_DEFECT = 'DEFECT'
CLASS_OK = 'OK'


def find_labels(labels_pth: List[str]) -> pd.DataFrame:
    # process label files first
    labels_df = pd.DataFrame()
    for label_pth in labels_pth:
        if not path.isfile(label_pth):
            raise FileNotFoundError(f'Could not find label file at '
                                    f'{label_pth}.')

        with open(label_pth) as f:
            label_df = pd.read_csv(f)
        label_df = label_df.drop('rating', axis=1)  # remove json column
        labels_df = pd.concat([labels_df, label_df])

    orig_len = len(labels_df)

    log.info(f'Registered {len(labels_df)} labels from {len(labels_pth)} '
             f'files.')

    return labels_df


def find_images(images_pth: List[str]) -> pd.DataFrame:
    # iterate over all image files to construct index
    images_idx = list()
    regex = re.compile(r'.*_(?P<uuid>.+)\.(png|jpg)')
    for image_pth in images_pth:
        if not path.isdir(image_pth):
            raise NotADirectoryError(f'{image_pth} is not a directory.')

        for f in os.listdir(image_pth):
            m = regex.match(f)

            if m:
                images_idx.append(
                    {'eventId': m.group('uuid'),
                     'filepath': path.abspath(path.join(image_pth, f))})

    images_df = pd.DataFrame(images_idx)
    log.info(f'Registered {len(images_idx)} images in {len(images_pth)} '
             f'directories.')

    return images_df


def write_pbtxt(classes: Dict[str, int], output_name: str):
    """
    Writes a labelmap .pbtxt file to disk.

    Parameters
    ----------
    classes: dict
        Mapping from string class name to integer class ID.
    output_name : str
        Path to file to write to.
    """
    with open(output_name, 'w') as f:
        for name, id_ in classes.items():
            out = ''
            out += 'item {\n'
            out += f'  id: {id_ + 1}\n'
            out += f'  name: \'{name}\'\n'
            out += '}\n\n'

            f.write(out)