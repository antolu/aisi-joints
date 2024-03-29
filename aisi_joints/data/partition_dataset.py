"""
This module provides functions to partition a dataset into train/validation(/test)
splits.

The partitioning aims to have equal numbers of samples from each platform
in each split.

This module is runnable. Use the `-h` option to view usage.
"""
import logging
import re
from argparse import ArgumentParser, Namespace
import pandas as pd
import numpy as np
import os.path as path

log = logging.getLogger(__name__)


def partition_dataset(df: pd.DataFrame, ratio: str):
    m = re.match(
        r'(?P<train>\d+)\/(?P<validation>\d+)(\/(?P<test>\d+))?', ratio
    )

    if not m:
        raise ValueError(f'Invalid ratio format: {ratio}.')

    train_ratio = int(m.group('train'))
    val_ratio = int(m.group('validation'))

    if m.group('test'):
        test_ratio = int(m.group('test'))
    else:
        test_ratio = 0

    # normalize ratios to sum to 100
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum != 100:
        log.warning(f'Sum of ratios is not 100, will normalize.')

        normalize = 100 / ratio_sum

        train_ratio *= normalize
        val_ratio *= normalize
        test_ratio *= normalize

    log.info(
        'Processing with ratio train/validation/test '
        f'{train_ratio}/{val_ratio}/{test_ratio}'
    )

    df['split'] = np.nan

    if 'platformId' in df.columns:
        platform_id = 'platformId'
    elif 'platformID' in df.columns:
        platform_id = 'platformID'
    else:
        raise ValueError()

    # split equally between platformIDs
    platforms = list(pd.unique(df[platform_id]))
    platform_dfs = []

    for platform in platforms:
        platform_df = df[df[platform_id] == platform]

        if test_ratio == 0:
            split = [round(train_ratio / 100 * len(platform_df))]
            split_dfs = np.split(platform_df.sample(frac=1), split)
        else:
            split = np.cumsum(
                [
                    round(train_ratio / 100 * len(platform_df)),
                    round(val_ratio / 100 * len(platform_df)),
                ]
            )
            split_dfs = np.split(platform_df.sample(frac=1), split)

        # assign the split label
        split_names = ('train', 'validation', 'test')

        for i, split_df in enumerate(split_dfs):
            split_df['split'] = split_names[i]

        platform_df = pd.concat(split_dfs)
        platform_dfs.append(platform_df)

    # join back together to write output
    df = pd.concat(platform_dfs)
    msg = 'Produced partitioned dataset with the following splits: '
    msg += f'{len(df[df["split"] == "train"])} train samples, '
    msg += f'{len(df[df["split"] == "validation"])} validation samples'

    if test_ratio != 0:
        msg += f', {len(df[df["split"] == "test"])} test samples.'
    else:
        msg += '.'
    log.info(msg)

    return df


def main(args: Namespace):
    with open(args.input_csv) as f:
        df = pd.read_csv(f)

    df = partition_dataset(df, args.ratio)

    df.to_csv(args.output, index=False)
    log.info(f'Wrote partitioned dataset to {path.abspath(args.output)}')


if __name__ == '__main__':
    from .._utils.logging import setup_logger

    parser = ArgumentParser()

    parser.add_argument(
        '-r',
        '--ratio',
        type=str,
        default='80/20',
        help='Split ratio between train/validation(/test)',
    )
    parser.add_argument(
        'input_csv',
        type=str,
        help='Name of unsplit or previously split dataset '
        'csv generated by preprocess_csv.',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output_split.csv',
        help='Name of split dataset csv.',
    )

    args = parser.parse_args()

    setup_logger()
    main(args)
