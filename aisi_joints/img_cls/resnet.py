import logging
from argparse import Namespace, ArgumentParser
import datetime
from functools import partial
from os import path
from typing import Tuple

import tensorflow as tf
import yaml
from keras import Model, Input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd

from aisi_joints.data.generate_tfrecord import read_tfrecord
from aisi_joints.img_cls.data import process_example, load_tfrecord
from aisi_joints.img_cls.evaluate import EvaluateImages
from aisi_joints.utils.logging import setup_logger

log = logging.getLogger(__name__)


def get_model() -> Tuple[Model, Model]:

    base_model: Model = tf.keras.applications.InceptionResNetV2(
        include_top=False, weights='imagenet', input_tensor=Input(shape=(299, 299, 3)))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model


def main(args: Namespace, config: dict):
    train_data = load_tfrecord(config['train_data'], config['batch_size'], random_crop=False)
    val_data = load_tfrecord(config['validation_data'], config['batch_size'], shuffle=False, random_crop=False)

    base_model, model = get_model()
    base_model.trainable = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    evaluate_images = EvaluateImages(model, config['validation_data'], config['batch_size'])
    img_writer = tf.summary.SummaryWriter(path.join(args.logdir, 'images'))

    # ========================================================================
    # Define callbacks
    # ========================================================================
    evaluate_images_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch: evaluate_images.evaluate(epoch, tb_writer=img_writer))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path.join(args.checkpoint_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # ========================================================================
    # Train top layers
    # ========================================================================
    base_model.trainable = False
    metrics = [tf.keras.metrics.Accuracy()]
    params = config['transfer']

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        params['learning_rate'], 25, 0.94, staircase=True
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                  loss=tf.keras.losses.CategoricalCrossEntropy(),
                  metrics=metrics)

    transfer_cb = [
        tf.keras.callbacks.TensorBoard(
            log_dir=path.join(args.logdir, str(timestamp), 'transfer')),
        evaluate_images_cb,
        model_checkpoint_callback,
    ]
    # train the model on the new data for a few epochs
    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=params['epochs'],
              callbacks=transfer_cb)

    # ========================================================================
    # Fine tuning of entire model
    # ========================================================================
    base_model.trainable = True

    params = config['finetune']
    finetune_lr = tf.keras.optimizers.schedule.CosineDecay(params['learning_rate'],
                                                           decay_steps=params['decay_steps'],
                                                           alpha=1e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
                  loss=tf.keras.losses.CategoricalCrossEntropy(),
                  metrics=metrics)

    finetune_cb = [
        tf.keras.callbacks.TensorBoard(
            log_dir=params.join(args.logdir, str(timestamp), 'finetune')),
        evaluate_images_cb,
        model_checkpoint_callback,
    ]
    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=params['metrics'],
              callbacks=finetune_cb)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.yml')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug logs')
    parser.add_argument('-c', '--checkpoint-dir', default='checkpoints',
                        dest='checkpoint_dir',
                        help='Directory to save checkpoint files.')

    args = parser.parse_args()

    setup_logger(args.debug)

    with open(args.config) as f:
        config = yaml.full_load(f)
    main(args, config)
