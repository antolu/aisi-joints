import datetime
import logging
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
from os import path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from self_supervised import LinearClassifierMethod
from self_supervised.moco import SelfSupervisedMethod

log = logging.getLogger(__name__)


def train(dataset_path: str, checkpoint_dir: str, log_dir: str, config,
          mode: str):
    os.environ['DATA_PATH'] = dataset_path

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode in ('both', 'base'):
        params = config.model_params

        model = SelfSupervisedMethod(params)

        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir,
            f'model-base-{timestamp}' '-{epoch}-{step_train_loss:.2f}',
            monitor='valid_class_acc',
            save_top_k=5,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False)

        logger = TensorBoardLogger(
            path.join(log_dir, f'model-base-{timestamp}'))

        trainer = pl.Trainer(logger=logger, accelerator='auto',
                             callbacks=[checkpoint_callback],
                             max_epochs=params.max_epochs)

        trainer.fit(model)

    if mode in ('both', 'linear'):
        classifier_params = config.classifier_params
        linear_model = LinearClassifierMethod(classifier_params)

        # model loading
        if mode == 'both':
            linear_model.from_moco_checkpoint(
                checkpoint_callback.best_model_path)
        elif mode == 'linear':
            if path.isdir(checkpoint_dir):
                files = [path.join(checkpoint_dir, o)
                         for o in os.listdir(checkpoint_dir)
                         if o.endswith('.ckpt')]

                latest = max(files, key=path.getctime)
                log.info(f'Reading classifier checkpoint from {latest}.')
                linear_model.from_moco_checkpoint(latest)
            else:
                log.info(
                    f'Reading classifier checkpoint from {checkpoint_dir}.')
                linear_model.from_moco_checkpoint(checkpoint_dir)
                checkpoint_dir = path.split(checkpoint_dir)[0]

        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir,
            f'model-classifier-{timestamp} '
            '-{epoch}-{valid_loss:.2f}',
            monitor='valid_acc1',
            save_top_k=5,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False)

        logger = TensorBoardLogger(
            path.join(log_dir, f'model-classifier-{timestamp}'))

        trainer = pl.Trainer(logger=logger, accelerator='auto',
                             callbacks=[checkpoint_callback],
                             max_epochs=classifier_params.max_epochs,
                             auto_lr_find=True)

        trainer.fit(linear_model)


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
