import logging
import os
from argparse import ArgumentParser
from functools import partial
from os import path
from typing import Optional, List, Union

import tensorflow as tf

from ._config import Config
from ._dataloader import JointsSequence
from ._log_images import EvaluateImages
from ._models import ModelWrapper
from .._utils.logging import setup_logger
from .._utils.utils import TensorBoardTool, get_latest

log = logging.getLogger(__name__)


class ModelCheckpointWithFreeze(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, mw: ModelWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mw = mw

    def _save_model(self, epoch, batch, logs):
        self._mw.model.trainable = True
        super()._save_model(epoch, batch, logs)  # noqa

        self._mw.freeze()


def fit_model(mw: ModelWrapper, optimizer: tf.keras.optimizers.Optimizer,
              train_data: Union[tf.data.Dataset, tf.keras.utils.Sequence],
              val_data: Union[tf.data.Dataset, tf.keras.utils.Sequence],
              epochs: int, name: str,
              metrics: Optional[List[tf.keras.metrics.Metric]] = None):
    img_writer = tf.summary.create_file_writer(
        path.join(mw.config.log_dir, mw.config.timestamp, f'{name}/images'))
    image_eval = EvaluateImages(mw.model, val_data, img_writer,
                                10)
    tensorboard_img_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: image_eval.evaluate(epoch))

    checkpoint_class = partial(ModelCheckpointWithFreeze, mw=mw)

    model_checkpoint_callback = checkpoint_class(
        filepath=path.join(mw.config.checkpoint_dir,
                           'model-' + name + '.{epoch}-{val_loss:.2f}.h5'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    mw.model.compile(optimizer=optimizer,
                     loss=tf.keras.losses.CategoricalCrossentropy(),
                     metrics=metrics)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=path.join(mw.config.log_dir, mw.config.timestamp, name)),
        tensorboard_img_cb,
        model_checkpoint_callback,
    ]

    # train the model on the new data for a few epochs
    mw.model.fit(train_data, batch_size=mw.config.batch_size,
                 validation_data=val_data, use_multiprocessing=False,
                 workers=mw.config.workers, epochs=epochs,
                 callbacks=callbacks, class_weight=mw.config.class_weights)


def train(config: Config, mode: str):
    mw = ModelWrapper(config)
    model = mw.model
    input_size = model.input_shape[1:3]

    train_data = JointsSequence(config.dataset, 'train', *input_size,
                                batch_size=config.batch_size)
    val_data = JointsSequence(config.dataset, 'validation', *input_size,
                              random_crop=False, augment_data=False,
                              batch_size=config.batch_size)

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(class_id=0, name='precision_OK'),
               tf.keras.metrics.Precision(class_id=1, name='precision_DEFECT'),
               tf.keras.metrics.Recall(class_id=0, name='recall_OK'),
               tf.keras.metrics.Recall(class_id=1, name='recall_DEFECT')]

    # =========================================================================
    # Pre-training steps
    # =========================================================================

    if not path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def save_model(name: str):
        model.trainable = True
        model.save_weights(
            path.join(config.checkpoint_dir, f'model-{name}-last-model.h5'))

    # =========================================================================
    # Do training \o/
    # =========================================================================

    mw.freeze()
    interrupted = False

    if mode in ('both', 'transfer'):
        try:
            fit_model(mw, config.transfer_config.optimizer, train_data,
                      val_data, config.transfer_config.epochs,
                      'transfer', metrics=metrics)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            save_model('transfer')

            if interrupted:
                raise KeyboardInterrupt

    model.trainable = True
    if mode in ('both', 'finetune'):
        if mode == 'finetune':
            # load model from checkpoint
            checkpoint_path = get_latest(config.checkpoint_dir,
                                         lambda o: o.endswith('.h5'))
            log.info(f'Loading checkpoint from {checkpoint_path}.')
            model.load_weights(checkpoint_path)

        mw.freeze_mode = 'partial'
        mw.freeze()
        try:
            fit_model(mw, config.finetune_config.optimizer, train_data,
                      val_data, config.finetune_config.epochs,
                      'finetune', metrics=metrics)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            save_model('finetune')

            if interrupted:
                raise KeyboardInterrupt


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')
    parser.add_argument('-d', '--dataset', help='Path to dataset .csv',
                        required=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Launch tensorboard as part of the script.')
    parser.add_argument('-c', '--checkpoint-dir', default='checkpoints',
                        dest='checkpoint_dir',
                        help='Directory to save checkpoint files.')
    parser.add_argument('-m', '--mode',
                        choices=['transfer', 'finetune', 'both'],
                        default='both',
                        help='Train classification layers, full CNN or both')

    args = parser.parse_args(argv)

    setup_logger()
    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    config = Config(args.config.replace('/', '.'))
    config.log_dir = args.logdir
    config.checkpoint_dir = args.checkpoint_dir
    if args.dataset is not None:
        config.dataset = args.dataset
    else:
        if config.dataset is None:
            raise ValueError('Must supply either config.dataset or '
                             'dataset command line argument.')

    if args.tensorboard:
        tensorboard = TensorBoardTool(config.log_dir)
        tensorboard.run()

    try:
        train(config, args.mode)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
