"""
This module provides a means ot compute the mean and standard deviation of
all images in a .csv dataset.

This module is runnable. Use the `-h` option to view usage.
"""
import logging
from argparse import Namespace, ArgumentParser

import numpy as np
import pandas as pd
from PIL import Image

from .._utils.logging import setup_logger

log = logging.getLogger(__name__)


def compute_mean_std(df: pd.DataFrame):
    if 'filepath' not in df.columns:
        msg = 'Could not find filepath column in dataframe'
        raise ValueError(msg)

    partial_mean = np.zeros(3)
    partial_sq = np.zeros(3)

    pixel_count = 0

    for file in df['filepath']:
        image = Image.open(file)
        arr = np.asarray(image) / 255
        pixel_count += image.width * image.height

        partial_mean += arr.mean(axis=(0, 1))

    mean = partial_mean / len(df)

    for file in df['filepath']:
        image = Image.open(file)
        arr = np.asarray(image) / 255

        partial_sq += np.sum((arr - mean) ** 2, axis=(0, 1)) / (
                    image.width * image.height)

    std = np.sqrt(partial_sq / len(df))

    return mean, std


def main(args: Namespace):
    df = pd.read_csv(args.csv_path)

    mean, std = compute_mean_std(df)

    log.info(f'Computed mean :{mean} and std: {std}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('csv_path',
                        help='Path to dataset .csv.')

    args = parser.parse_args()

    setup_logger()
    main(args)
