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
from aisi_joints.img_cls.data import get_data, process_example

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

    freeze_layers(base_model)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return base_model, model


def freeze_layers(model: Model):
    for layer in model.layers:
        layer.trainable = False


def main(config: dict):
    train_data = tf.data.TFRecordDataset([config['train_data']])
    val_data = tf.data.TFRecordDataset([config['validation_data']])

    train_data = train_data.map(partial(process_example), tf.data.AUTOTUNE).shuffle(2048, reshuffle_each_iteration=True).batch(config['batch_size'])
    val_data = val_data.map(partial(process_example), tf.data.AUTOTUNE).shuffle(2048, reshuffle_each_iteration=True).batch(config['batch_size'])

    base_model, model = get_model()

    # train the model on the new data for a few epochs
    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=config['epochs'])

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

    model.fit(train_data, batch_size=config['batch_size'], validation_data=val_data,
              use_multiprocessing=True, workers=config['workers'],
              epochs=config['epochs'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.yml')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.full_load(f)
    main(config)
