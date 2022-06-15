import datetime
import logging
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
from os import path
from typing import Optional, List

import attr
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import AttributeDict

from self_supervised import LinearClassifierMethod, ModelParams, \
    LinearClassifierMethodParams
from self_supervised.moco import SelfSupervisedMethod
from .._utils.utils import get_latest

log = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')


def train_encoder(params: ModelParams, checkpoint_dir: str,
                  log_dir: str, timestamp: Optional[str] = None,
                  callbacks: Optional[Callback] = None) \
        -> ModelCheckpoint:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model = SelfSupervisedMethod(params)

    callbacks = [] if callbacks is None else callbacks
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir,
            f'model-base-{timestamp}' '-{epoch}-{step_train_loss:.2f}',
            monitor='valid_class_acc',
            save_top_k=3,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    if log_dir is not None:
        logger = TensorBoardLogger(
            path.join(log_dir, f'model-base-{timestamp}'))
    else:
        logger = True

    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)
    trainer = pl.Trainer(logger=logger, accelerator='auto',
                         callbacks=callbacks,
                         max_epochs=params.max_epochs)

    trainer.fit(model)

    return checkpoint_callback


def train_classifier(params: LinearClassifierMethodParams,
                     checkpoint_path: str,
                     checkpoint_dir: str, log_dir: str,
                     timestamp: Optional[str] = None,
                     callbacks: Optional[List[Callback]] = None) \
        -> ModelCheckpoint:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    params_dict = AttributeDict(attr.asdict(params))
    params_dict.pop('encoder_arch')
    params_dict.pop('embedding_dim')
    params_dict.pop('dataset_name')
    model = LinearClassifierMethod.from_moco_checkpoint(
        checkpoint_path, **params_dict)

    callbacks = callbacks if callbacks is not None else []
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir,
            f'model-classifier-{timestamp}'
            '-{epoch}-{valid_loss:.2f}',
            monitor='valid_acc1',
            save_top_k=3,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    if log_dir is not None:
        logger = TensorBoardLogger(
            path.join(log_dir, f'model-classifier-{timestamp}'))
    else:
        logger = True

    trainer = pl.Trainer(logger=logger, accelerator='auto',
                         callbacks=callbacks,
                         max_epochs=params.max_epochs)

    trainer.fit(model)

    return checkpoint_callback


def train(dataset_path: str, checkpoint_dir: str, log_dir: str, config,
          mode: str):
    os.environ['DATA_PATH'] = dataset_path

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode in ('both', 'base'):
        params = config.model_params

        model_checkpoint = train_encoder(config.model_params, checkpoint_dir,
                                         log_dir, timestamp)

    if mode in ('both', 'linear'):
        classifier_params: LinearClassifierMethodParams = config \
            .classifier_params
        linear_model = LinearClassifierMethod(classifier_params)

        checkpoint_path = ''
        # model loading
        if mode == 'both':
            checkpoint_path = model_checkpoint.best_model_path
        elif mode == 'linear':
            checkpoint_path = get_latest(checkpoint_dir,
                                         lambda o: o.endswith('.ckpt'))
        else:
            raise NotImplementedError

        model_checkpoint = train_classifier(classifier_params, checkpoint_path,
                                            checkpoint_dir, log_dir, timestamp)


def main(args: Namespace, config):
    train(args.dataset, args.checkpoint_dir, args.logdir, config, args.mode)


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
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')

    args = parser.parse_args()

    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    config_module = import_module(args.config.replace('/', '.'))

    main(args, config_module)
