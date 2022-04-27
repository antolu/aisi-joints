import logging
from argparse import ArgumentParser, Namespace
from typing import List

import pandas as pd


log = logging.getLogger(__name__)


def concatenate_dataset(dfs: List[pd.DataFrame]) -> pd.DataFrame:

    cleaned = []

    for df in dfs:
        df = df.copy()

        unnamed = [o for o in df.columns if o.startswith('Unnamed')]
        if len(unnamed) > 0:
            df.drop(columns=unnamed, inplace=True)

        cleaned.append(df)

    return pd.concat(cleaned)


def main(args: Namespace):
    dfs = []
    for file in args.input:
        dfs.append(pd.read_csv(file))

    concatenated = concatenate_dataset(dfs)

    concatenated.to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('input', nargs='+',
                        help='Input .csv datasets to concatenate.')
    parser.add_argument('-o', '--output', default='concatenated.csv',
                        help='Path to output concatenated .csv')

    args = parser.parse_args()

    main(args)


