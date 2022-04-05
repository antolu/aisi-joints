import logging
from argparse import Namespace, ArgumentParser
from functools import partial
from typing import Tuple

import tensorflow as tf
import yaml
from keras import Model, Input
from keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import SGD
import pandas as pd

from aisi_joints.data.generate_tfrecord import read_tfrecord
from aisi_joints.img_cls.data import process_example, load_tfrecord
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
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model


def freeze_layers(model: Model):
    for layer in model.layers:
        layer.trainable = False


def main(args: Namespace, config: dict):
    train_data = load_tfrecord(config['train_data'], config['batch_size'])
    val_data = load_tfrecord(config['validation_data'], config['batch_size'], shuffle=False)

    base_model, model = get_model()
    base_model.trainable = False

    transfer_cb = tf.keras.callbacks.TensorBoard(log_dir=args.logdir + '/transfer')
    finetune_cb = tf.keras.callbacks.TensorBoard(log_dir=args.logdir + '/finetune')

    # tf_writer = tf.summary.create_file_writer(args.logdir)

    # def images_callback():
    #     with tf_writer.as_default():
    #         tf.summary.image("Training data", img, step=0)

    metrics = [tf.keras.metrics.Accuracy(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
                  metrics=metrics)

    # train the model on the new data for a few epochs
    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=config['epochs'],
              callbacks=[transfer_cb])

    base_model.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy',
                  metrics=metrics)

    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=config['epochs'],
              callbacks=[finetune_cb])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.yml')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug logs')

    args = parser.parse_args()

    setup_logger(args.debug)

    with open(args.config) as f:
        config = yaml.full_load(f)
    main(args, config)
