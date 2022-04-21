import logging
import os
from argparse import ArgumentParser
from os import path

import tensorflow as tf

from .config import Config
from .models import get_model
from ..utils.logging import setup_logger

log = logging.getLogger(__name__)


def export_model(config: Config, checkpoint_dir: str, output_dir: str):
    base_model, model = get_model(config.base_model)

    if path.isdir(checkpoint_dir):
        files = [path.join(checkpoint_dir, o)
                 for o in os.listdir(checkpoint_dir) if o.endswith('.h5')]

        latest = max(files, key=path.getctime)
        log.info(f'Reading checkpoint from {latest}.')
        model.load_weights(latest)
    else:
        log.info(f'Reading checkpoint from {checkpoint_dir}.')
        model.load_weights(checkpoint_dir)

    log.info(f'Writing model to {output_dir}.')
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    log.info('Done!')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('config', help='Path to config.yml')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir',
                        default='checkpoints', type=str,
                        help='Path to directory with trained checkpoints. '
                             'Will use latest checkpoint from the directory '
                             'unless a specific file is selected.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str,
                        default='exported_models',
                        help='Directory to save exported model to.')

    args = parser.parse_args()
    setup_logger()

    conf = Config(args.config)

    export_model(conf, args.checkpoint_dir, args.output_dir)
