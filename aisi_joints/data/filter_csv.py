"""
Script to remove rows of a dataset .csv based on eventIds, where the eventIds
to remove are in a second .csv.

This module is runnable. Use the `-h` option to view usage.
"""
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import pandas as pd

import logging
log = logging.getLogger(__name__)


def filter_dataframe(df: pd.DataFrame, filters: List[pd.DataFrame]) \
        -> Tuple[pd.DataFrame, str]:
    msgs = []
    for i, filter in enumerate(filters):
        orig_len = len(df)

        df = df[~df[['eventId']].apply(tuple, 1).isin(filter[['eventId']].apply(tuple, 1))]

        msg = f'Removed {orig_len - len(df)} samples on pass {i + 1}.'
        msgs.append(msg)
        log.info(msg)

    msg = f'Remaining number of samples: {len(df)}.'
    msgs.append(msg)
    log.info(msg)

    return df, '\n'.join(msgs)


def main(args: Namespace):

    df = pd.read_csv(args.input_csv)

    filters = []
    for csv in args.filter:
        filters.append(pd.read_csv(csv))

    df, _ = filter_dataframe(df, filters)

    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    from .._utils.logging import setup_logger
    parser = ArgumentParser()

    parser.add_argument('input_csv', type=str,
                        help='Name of unsplit or previously split dataset '
                             'csv generated by preprocess_csv.')
    parser.add_argument('-f', '--filter', nargs='+',
                        help='.csv files with eventIds to remove.')
    parser.add_argument('-o', '--output', type=str,
                        default='output_updated.csv',
                        help='Name of split dataset csv.')

    args = parser.parse_args()

    setup_logger()
    main(args)
