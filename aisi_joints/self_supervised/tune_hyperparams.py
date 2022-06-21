import datetime
import logging
import os
from argparse import ArgumentParser, Namespace
from copy import copy
from importlib import import_module
from pprint import pformat
from typing import Union

from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from self_supervised import ModelParams, LinearClassifierMethodParams
from .train import train_encoder, train_classifier
from .._utils import TensorBoardTool, setup_logger, get_latest

log = logging.getLogger(__name__)


def tune_encoder(config: dict, params: ModelParams,
                 **kwargs) -> Union[None, ModelCheckpoint]:
    params = copy(params)

    for k, v in config.items():
        setattr(params, k, v)

    if 'log_dir' in kwargs:
        log_dir = kwargs.pop('log_dir')
    else:
        log_dir = tune.get_trial_dir()

    return train_encoder(params, log_dir=log_dir, **kwargs)


def tune_classifier(config: dict, params: LinearClassifierMethodParams,
                    **kwargs) -> Union[None, ModelCheckpoint]:
    params = copy(params)

    for k, v in config.items():
        setattr(params, k, v)

    if 'log_dir' in kwargs:
        log_dir = kwargs.pop('log_dir')
    else:
        log_dir = tune.get_trial_dir()

    return train_classifier(params, log_dir=log_dir, **kwargs)


def main(args: Namespace, config):
    os.environ['DATA_PATH'] = os.path.abspath(args.dataset)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    resources_per_trial = {'cpu': 4, 'gpu': 1}

    if args.mode in ('both', 'base'):
        params: ModelParams = config.model_params

        grid = {
            'mlp_hidden_dim': tune.choice([512, 1024, 2048, 3072, 4096]),
            'dim': tune.choice([1024, 2048, 3072, 4096]),
            'lr': tune.loguniform(1.e-4, 1),
            'weight_decay': tune.loguniform(1.e-5, 1.e-1),
            'momentum': tune.uniform(0.9, 1.0),
            'lars_eta': tune.loguniform(1.e-4, 1.e-1),
            "batch_size": tune.choice([32, 64, 128]),
        }

        scheduler = ASHAScheduler(
            time_attr='epoch',
            max_t=25,
            grace_period=5,
            reduction_factor=2
        )

        reporter = CLIReporter(
            parameter_columns=['lr', 'weight_decay', 'momentum',
                               'mlp_hidden_dim', 'dim', 'lars_eta'],
            metric_columns=['loss', 'val_accuracy', 'epoch'])

        callback = TuneReportCallback(
            {
                'loss': 'step_train_loss',
                'val_accuracy': 'valid_class_acc',
                'epoch': 'epoch'
            },
            on='validation_end')

        train_fn_with_parameters = tune.with_parameters(
            tune_encoder,
            params=params,
            checkpoint_dir=None,
            timestamp=None,
            callbacks=[callback],
        )

        analysis = tune.run(train_fn_with_parameters,
                            metric='loss',
                            mode='min',
                            config=grid,
                            num_samples=256,
                            scheduler=scheduler,
                            progress_reporter=reporter,
                            resources_per_trial=resources_per_trial,
                            local_dir=f'{args.logdir}/ray_results',
                            log_to_file=True,
                            name='tune_encoder')

        log.info('Best hyperparameters found were: ', analysis.best_config)
        log.info(f'All results: \n' + pformat(analysis.results))

        log.info('Training model with tuned parameters.')

        checkpoint = tune_encoder(analysis.best_config, params,
                                  checkpoint_dir=args.checkpoint_dir,
                                  log_dir=args.logdir,
                                  timestamp=timestamp)

    if args.mode in ('both', 'linear'):
        params: LinearClassifierMethodParams = config.classifier_params
        if args.mode == 'both':
            checkpoint_path = checkpoint.best_model_path
        else:
            checkpoint_path = get_latest(
                args.checkpoint_dir, lambda o: o.startswith('model-base')
                                               and o.endswith('.ckpt'))

        grid = {
            'lr': tune.loguniform(1.e-4, 1),
            'weight_decay': tune.loguniform(1.e-5, 1.e-1),
            'momentum': tune.uniform(0.6, 1.0),
            "batch_size": tune.choice([32, 64, 128, 256]),
        }

        scheduler = ASHAScheduler(
            time_attr='epoch',
            max_t=50,
            grace_period=5,
            reduction_factor=2
        )

        reporter = CLIReporter(
            parameter_columns=['lr', 'weight_decay', 'momentum',
                               'mlp_hidden_dim', 'dim', 'lars_eta'],
            metric_columns=['loss', 'val_accuracy', 'epoch']
        )

        callback = TuneReportCallback(
            {
                'loss': 'step_train_loss',
                'val_accuracy': 'valid_acc1',
                'epoch': 'epoch'
            },
            on='validation_end'
        )

        train_fn_with_parameters = tune.with_parameters(
            tune_classifier,
            params=params,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=None,
            timestamp=None,
            callbacks=[callback],
        )

        analysis = tune.run(train_fn_with_parameters,
                            metric='loss',
                            mode='min',
                            config=grid,
                            num_samples=-1,
                            scheduler=scheduler,
                            progress_reporter=reporter,
                            resources_per_trial=resources_per_trial,
                            local_dir=f'{args.logdir}/ray_results',
                            log_to_file=True,
                            name='tune_classifier')

        log.info('Best hyperparameters found were: ', analysis.best_config)
        log.info(f'All results: \n' + pformat(analysis.results))

        log.info('Training classifier with tuned parameters.')

        checkpoint = tune_classifier(analysis.best_config, params,
                                     checkpoint_path=checkpoint_path,
                                     checkpoint_dir=args.checkpoint_dir,
                                     log_dir=args.logdir,
                                     timestamp=timestamp)


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

    setup_logger(file_logger=f'{args.logdir}/log.log')
    main(args, config_module)
