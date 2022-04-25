import logging
import os
import sys
from argparse import ArgumentParser
from os import path
from typing import Optional, List

import tensorflow as tf
from keras import Model
from tensorboard import program, default

from ._config import Config, FitConfig
from ._dataloader import load_tfrecord
from ._log_images import EvaluateImages
from ._models import get_model
from ..utils.logging import setup_logger

log = logging.getLogger(__name__)


def fit_model(model: Model, optimizer: tf.keras.optimizers.Optimizer,
              train_data: tf.data.Dataset, val_data: tf.data.Dataset,
              config: Config, fit_config: FitConfig, name: str,
              metrics: Optional[List[tf.keras.metrics.Metric]] = None):
    img_writer = tf.summary.create_file_writer(
        path.join(config.log_dir, config.timestamp, f'{name}/images'))
    image_eval = EvaluateImages(model, config.validation_data, img_writer,
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
              workers=config.workers, epochs=fit_config.epochs,
              callbacks=callbacks, class_weight=config.class_weights)


def main(config: Config):
    train_data = load_tfrecord(config.train_data, config.bs, random_crop=True)
    val_data = load_tfrecord(config.validation_data, config.bs, shuffle=False,
                             random_crop=False, augment_data=False)

    base_model, model = get_model(config.base_model)

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
        model.save_weights(
            path.join(config.checkpoint_dir, f'{name}_latest_model.h5'))

    # =========================================================================
    # Do training \o/
    # =========================================================================

    base_model.trainable = False

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            config.transfer_config.learning_rate,
            config.transfer_config.decay_steps, 0.94, staircase=True
        )
    )
    try:
        fit_model(model, optimizer, train_data, val_data,
                  config, config.transfer_config, 'transfer', metrics=metrics)
    except KeyboardInterrupt:
        save_model('transfer')
        raise

    base_model.trainable = True

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            config.finetune_config.lr,
            decay_steps=config.finetune_config.decay_steps, alpha=1e-6
        )
    )
    try:
        fit_model(model, optimizer, train_data, val_data,
                  config, config.finetune_config, 'finetune', metrics=metrics)
    except KeyboardInterrupt:
        save_model('finetune')
        raise


class TensorBoardTool:

    def __init__(self, log_dir: str):
        self.log_dir: str = log_dir

    def run(self):
        if not path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Suppress http messages
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins())
        tb.configure(argv=[None, '--logdir', self.log_dir])

        url = tb.launch()
        sys.stdout.write(f'TensorBoard at {url} \n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.yml')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug logs')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Launch tensorboard as part of the script.')
    parser.add_argument('-c', '--checkpoint-dir', default='checkpoints',
                        dest='checkpoint_dir',
                        help='Directory to save checkpoint files.')

    args = parser.parse_args()

    setup_logger(args.debug)
    config = Config(args.config)
    config.log_dir = args.logdir
    config.checkpoint_dir = args.checkpoint_dir

    if args.tensorboard:
        tensorboard = TensorBoardTool(config.log_dir)
        tensorboard.run()

    try:
        main(config)
    except KeyboardInterrupt:
        pass
