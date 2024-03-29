"""
This module provides everything necessary to export a trained model.

This module is runnable. Use the `-h` option to view usage.
"""
import logging
import os
from argparse import ArgumentParser
from typing import List, Optional

from ._config import Config
from ._models import ModelWrapper
from .._utils import get_latest
from .._utils.logging import setup_logger

log = logging.getLogger(__name__)


def export_model(config: Config, checkpoint_dir: str, output_dir: str):
    """
    Construct model from config and load checkpoint weights onto it, then
    export it.

    Parameters
    ----------
    config: Config
    checkpoint_dir: str
    output_dir: str
    """
    model = ModelWrapper(config).model

    checkpoint_path = get_latest(checkpoint_dir, lambda o: o.endswith('.h5'))
    log.info(f'Reading checkpoint from {checkpoint_path}.')
    model.load_weights(checkpoint_path, by_name=True)

    log.info(f'Writing model to {output_dir}.')
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    log.info('Done!')


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()

    parser.add_argument('config', help='Path to config.py')
    parser.add_argument(
        '-m',
        '--checkpoint-dir',
        dest='checkpoint_dir',
        default='checkpoints',
        type=str,
        help='Path to directory with trained checkpoints. '
        'Will use latest checkpoint from the directory '
        'unless a specific file is selected.',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='output_dir',
        type=str,
        default='exported_models',
        help='Directory to save exported model to.',
    )

    args = parser.parse_args(argv)
    setup_logger()

    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    conf = Config(args.config.replace('/', '.'))

    export_model(conf, args.checkpoint_dir, args.output_dir)


if __name__ == '__main__':
    main()
