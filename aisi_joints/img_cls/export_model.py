import logging
import os
from argparse import ArgumentParser
from typing import List, Optional

from ._config import Config
from ._models import get_model
from .._utils import get_latest
from .._utils.logging import setup_logger

log = logging.getLogger(__name__)


def export_model(config: Config, checkpoint_dir: str, output_dir: str):
    base_model, model, _ = get_model(config.base_model, config.fc_hidden_dim,
                                     config.fc_dropout, config.fc_num_layers,
                                     config.fc_activation)

    checkpoint_path = get_latest(checkpoint_dir, lambda o: o.endswith('.h5'))
    log.info(f'Reading checkpoint from {checkpoint_path}.')
    model.load_weights(checkpoint_path)

    log.info(f'Writing model to {output_dir}.')
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    log.info('Done!')


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()

    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir',
                        default='checkpoints', type=str,
                        help='Path to directory with trained checkpoints. '
                             'Will use latest checkpoint from the directory '
                             'unless a specific file is selected.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str,
                        default='exported_models',
                        help='Directory to save exported model to.')

    args = parser.parse_args(argv)
    setup_logger()

    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    conf = Config(args.config.replace('/', '.'))

    export_model(conf, args.checkpoint_dir, args.output_dir)


if __name__ == '__main__':
    main()
