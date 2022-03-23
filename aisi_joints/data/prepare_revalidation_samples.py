import os
import shutil
from argparse import ArgumentParser, Namespace
from os import path

import pandas as pd
import logging

log = logging.getLogger(__name__)


def export_revalidation_samples(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)

    df = df.drop(columns=['cls', 'x0', 'x1', 'y0', 'y1'])

    if not path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(path.join(output_dir, 'samples.csv'))

    # copy images
    for row in df.itertuples():
        shutil.copy(row.filepath,
                    path.join(output_dir, path.split(row.filepath)[-1]))

    log.info(f'Copied {len(df)} images to output directory {output_dir} '
             f'and created samples.csv.')


if __name__ == '__main__':
    from ..utils.logging import setup_logger
    parser = ArgumentParser()
    parser.add_argument('csv', type=str,
                        help='Path to .csv file containing samples to be re-evaluated.')
    parser.add_argument('-o', '--output', type=str, default='to_reevaluate',
                        help='Directory to path where images requiring re-evaluation will copied to.')

    args = parser.parse_args()

    setup_logger()
    export_revalidation_samples(args.csv, args.output)
