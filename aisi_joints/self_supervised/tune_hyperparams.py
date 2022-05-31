import datetime
import logging
import os
from argparse import ArgumentParser, Namespace
from copy import copy
from importlib import import_module

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from self_supervised import ModelParams
from .train import train_encoder
from .._utils.utils import TensorBoardTool

log = logging.getLogger(__name__)


def tune_encoder(config: dict, params: ModelParams,
                 *args, **kwargs):
    params = copy(params)

    for k, v in config.items():
        setattr(params, k, v)

    train_encoder(params, *args, log_dir=tune.get_trial_dir(), **kwargs)


def main(args: Namespace, config):
    os.environ['DATA_PATH'] = os.path.abspath(args.dataset)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.mode in ('both', 'base'):
        params: ModelParams = config.model_params

        grid = {
            'mlp_hidden_dim': tune.choice([512, 1024, 2048, 3072, 4096]),
            'dim': tune.choice([1024, 2048, 3072, 4096]),
            'lr': tune.loguniform(1.e-4, 1),
            'weight_decay': tune.loguniform(1.e-5, 1.e-1),
            'momentum': tune.uniform(0.6, 1.0),
            'lars_eta': tune.loguniform(1.e-4, 1.e-1),
            "batch_size": tune.choice([32, 64, 128]),
        }

        scheduler = ASHAScheduler(
            time_attr='epochs',
            max_t=25,
            grace_period=5,
            reduction_factor=2
        )

        reporter = CLIReporter(
            parameter_columns=['lr', 'weight_decay', 'momentum',
                               'mlp_hidden_dim', 'dim', 'lars_eta'],
            metric_columns=['loss', 'mean_accuracy', 'training_iteration'])

        callback = TuneReportCallback(
            {
                'loss': 'step_train_loss',
                'mean_accuracy': 'valid_class_acc'
            },
            on='validation_end')

        train_fn_with_parameters = tune.with_parameters(
            tune_encoder,
            params=params,
            checkpoint_dir=None,
            timestamp=None,
            callbacks=[callback],
        )

        resources_per_trial = {'cpu': 4, 'gpu': 1}

        analysis = tune.run(train_fn_with_parameters,
                            metric='loss',
                            mode='min',
                            config=grid,
                            num_samples=-1,
                            scheduler=scheduler,
                            progress_reporter=reporter,
                            resources_per_trial=resources_per_trial,
                            name='tune_encoder')

        print('Best hyperparameters found were: ', analysis.best_config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-d', '--dataset', help='Path to dataset .csv',
                        required=True)
    parser.add_argument('-c', '--checkpoint-dir', dest='checkpoint_dir',
                        help='Path to checkpoint dir.', default='checkpoints')
    parser.add_argument('-m', '--mode', choices=['both', 'base', 'linear'],
                        default='both',
                        help='Train base encoder model, linear classifier '
                             'or both.')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Launch tensorboard as part of the script.')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')

    args = parser.parse_args()

    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    config_module = import_module(args.config.replace('/', '.'))

    if args.tensorboard:
        tensorboard = TensorBoardTool(args.logdir)
        tensorboard.run()

    main(args, config_module)
