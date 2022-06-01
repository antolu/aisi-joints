import logging
import os
from argparse import ArgumentParser
from os import path
from typing import Optional, List, Union

import tensorflow as tf
from keras import Model

from .._utils.utils import TensorBoardTool
from ._config import Config
from ._dataloader import prepare_dataset, JointsSequence
from ._log_images import EvaluateImages
from ._models import get_model
from .._utils.logging import setup_logger

log = logging.getLogger(__name__)


def fit_model(model: Model, optimizer: tf.keras.optimizers.Optimizer,
              train_data: Union[tf.data.Dataset, tf.keras.utils.Sequence],
              val_data: Union[tf.data.Dataset, tf.keras.utils.Sequence],
              config: Config, epochs: int, name: str,
              metrics: Optional[List[tf.keras.metrics.Metric]] = None):
    img_writer = tf.summary.create_file_writer(
        path.join(config.log_dir, config.timestamp, f'{name}/images'))
    image_eval = EvaluateImages(model, config.dataset, img_writer,
                                config.bs, 10)
    tensorboard_img_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: image_eval.evaluate(epoch))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path.join(config.checkpoint_dir,
                           'model.{epoch:02d}-{val_loss:.2f}.h5'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=metrics)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=path.join(config.log_dir, config.timestamp, name)),
        tensorboard_img_cb,
        model_checkpoint_callback,
    ]

    # train the model on the new data for a few epochs
    model.fit(train_data, batch_size=config.batch_size,
              validation_data=val_data, use_multiprocessing=True,
              workers=config.workers, epochs=epochs,
              callbacks=callbacks, class_weight=config.class_weights)


def main(config: Config):
    base_model, model, _ = get_model(config.base_model, config.fc_hidden_dim,
                                     config.fc_dropout)
    input_size = base_model.input_shape[1:3]

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

    if not path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_model(name: str):
        model.trainable = False
        model.save_weights(
            path.join(config.checkpoint_dir, f'{name}_last_model.h5'))

    # =========================================================================
    # Do training \o/
    # =========================================================================

    base_model.trainable = False

    try:
        fit_model(model, config.transfer_config.optimizer, train_data,
                  val_data, config, config.transfer_config.epochs, 'transfer',
                  metrics=metrics)
    except KeyboardInterrupt:
        save_model('transfer')
        raise

    base_model.trainable = True

    try:
        fit_model(model, config.finetune_config.optimizer, train_data,
                  val_data, config, config.finetune_config.epochs, 'finetune',
                  metrics=metrics)
    except KeyboardInterrupt:
        save_model('finetune')
        raise


if __name__ == '__main__':
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

    args = parser.parse_args()

    setup_logger(args.debug)
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
        main(config)
    except KeyboardInterrupt:
        pass
