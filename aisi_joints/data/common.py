"""
This module provides common functions for data processing.
"""
import logging
import os
import re
from dataclasses import dataclass
from os import path
from PIL import Image
from typing import List, Dict

import pandas as pd

log = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.INFO)


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


def find_images(images_pth: List[str], find_dims: bool = True) \
        -> pd.DataFrame:
    """
    Walks through one or more directories (recursively) to find images,
    extracts eventId from the filename and maps the eventId to the image
    path using a Pandas DataFrame.

    By specifying `find_dims=True`, the function will also find the height
    and width of the images and put them into the dataframe.

    Parameters
    ----------
    images_pth: list
    find_dims: bool

    Returns
    -------
    pd.DataFrame
    """
    images_idx = list()

    # find images with a specific regex pattern only.
    regex = re.compile(r'.*_(?P<uuid>.+)\.(png|jpg)')
    for image_pth in images_pth:
        if not path.isdir(image_pth):
            raise NotADirectoryError(f'{image_pth} is not a directory.')

        for root, dirs, files in os.walk(image_pth):
            for f in files:
                m = regex.match(f)

                if not m:
                    continue

                images_idx.append(
                    {'eventId': m.group('uuid'),
                     'filepath': path.abspath(path.join(root, f))})

                if find_dims:
                    width, height = Image.open(path.join(root, f)).size
                    images_idx[-1].update({'width': width, 'height': height})

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
            out += f'  id: {id_}\n'
            out += f'  name: \'{name}\'\n'
            out += '}\n\n'

            f.write(out)


@dataclass
class DetectionBox:
    x0: int
    x1: int
    y0: int
    y1: int
    cls: str
    score: float = -1

    def to_coords(self) -> List[List[int]]:
        return [[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1],
                [self.x0, self.y1], [self.x0, self.y0]]

    def to_pascal_voc(self):
        return [self.x0, self.x1, self.y0, self.y1]


class Sample:
    eventId: str
    filepath: str
    bbox: DetectionBox

    has_detection: bool
    num_detections: int

    detected_bbox = List[DetectionBox]

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> 'Sample':
        sample = Sample()
        sample.eventId = df['eventId']
        sample.filepath = df['filepath']
        sample.bbox = DetectionBox(df['x0'], df['x1'], df['y0'], df['y1'],
                                   df['cls'])

        if 'detected_class' not in df.index:
            sample.has_detection = False
            return sample

        sample.has_detection = True
        sample.num_detections = df['num_detections']

        sample.detected_bbox = []
        x0 = [o for o in map(int, str(df['detected_x0']).split(';'))]
        x1 = [o for o in map(int, str(df['detected_x1']).split(';'))]
        y0 = [o for o in map(int, str(df['detected_y0']).split(';'))]
        y1 = [o for o in map(int, str(df['detected_y1']).split(';'))]
        scores = [o for o in map(float, str(df['detection_score']).split(';'))]
        cls = [o for o in df['detected_class'].split(';')]

        for i in range(sample.num_detections):
            sample.detected_bbox.append(
                DetectionBox(x0[i], x1[i], y0[i], y1[i], cls[i], scores[i]))

        return sample
