import logging
import os
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
from aisi_joints.img_cls.dataloader import process_example, load_tfrecord
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
    train_data = load_tfrecord(config['train_data'], config['batch_size'], random_crop=True)
    val_data = load_tfrecord(config['validation_data'], config['batch_size'], shuffle=False, random_crop=False, augment_data=False)

    base_model, model = get_model()
    base_model.trainable = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    img_writer_transfer = tf.summary.create_file_writer(path.join(args.logdir, timestamp, 'transfer/images'))
    img_writer_finetune = tf.summary.create_file_writer(path.join(args.logdir, timestamp, 'finetune/images'))
    evaluate_images_transfer = EvaluateImages(model, config['validation_data'], img_writer_transfer, config['batch_size'], 10)
    evaluate_images_finetune = EvaluateImages(model, config['validation_data'], img_writer_finetune, config['batch_size'], 10)

    # evaluate_images.evaluate(0, tb_writer=img_writer)

    # ========================================================================
    # Define callbacks
    # ========================================================================
    evaluate_images_transfer_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: evaluate_images_transfer.evaluate(epoch))
    evaluate_images_finetune_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: evaluate_images_finetune.evaluate(epoch))
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
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(class_id=0, name='precision_OK'),
               tf.keras.metrics.Precision(class_id=1, name='precision_DEFECT'),
               tf.keras.metrics.Recall(class_id=0, name='recall_OK'),
               tf.keras.metrics.Recall(class_id=1, name='recall_DEFECT')]
    params = config['transfer']

    transfer_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        params['learning_rate'], 25, 0.94, staircase=True
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=transfer_lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=metrics)

    log.info(f"Logging tensorboard to {path.join(args.logdir, timestamp, 'transfer')}")
    transfer_cb = [
        tf.keras.callbacks.TensorBoard(
            log_dir=path.join(args.logdir, timestamp, 'transfer')),
        evaluate_images_transfer_cb,
        model_checkpoint_callback,
    ]

    CLASS_WEIGHT = {0: 0.6, 1: 2.2}

    # train the model on the new data for a few epochs
    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=params['epochs'],
              callbacks=transfer_cb, class_weight=CLASS_WEIGHT)

    # ========================================================================
    # Fine tuning of entire model
    # ========================================================================
    base_model.trainable = True

    params = config['finetune']
    finetune_lr = tf.keras.optimizers.schedules.CosineDecay(params['learning_rate'],
                                                           decay_steps=params['decay_steps'],
                                                           alpha=1e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=metrics)

    finetune_cb = [
        tf.keras.callbacks.TensorBoard(
            log_dir=path.join(args.logdir, timestamp, 'finetune')),
        evaluate_images_finetune_cb,
        model_checkpoint_callback,
    ]
    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=params['epochs'],
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
