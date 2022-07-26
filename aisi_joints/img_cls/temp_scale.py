import argparse
import logging
import os
from os import path
from typing import List, Optional

import tensorflow as tf

from ._models import MLP
from ._config import Config
from ._dataloader import JointsSequence
from .._utils import setup_logger

log = logging.getLogger(__name__)


class TempScale(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self._temperature = tf.Variable(1.5, dtype="float32", trainable=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Input logits (pre softmax), output temperature scaled logits.

        Parameters
        ----------
        inputs : tf.Tensor

        Returns
        -------
        tf.Tensor
        """
        return inputs / self._temperature

    @property
    def temperature(self) -> tf.Tensor:
        return self._temperature.value()


def temp_scale(config: Config, save_dir: str, model_dir: str):
    model: tf.keras.Model = tf.keras.models.load_model(model_dir)
    input_size = model.input_shape[1:3]

    dataset = JointsSequence(config.dataset, 'validation', *input_size,
                             random_crop=False, augment_data=False,
                             batch_size=config.batch_size)

    input_ = model.input
    if model.layers[-1].name == 'model':
        classification_layer = model.layers[-1].layers[-1]
    else:
        if isinstance(model.layers[-1], MLP):
            classification_layer = model.layers[-1]
        else:
            raise ValueError('Don\'t know what to do.')

    if isinstance(classification_layer, MLP):
        last_dense = classification_layer._sublayers[-1]
    else:
        last_dense = classification_layer

    last_dense.activation = tf.keras.activations.linear
    model.trainable = False  # freeze all layers

    x = model.output
    temp_scale = TempScale()
    x = temp_scale(x)
    model = tf.keras.models.Model(input_, x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1)
    nll_criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True
    )

    model.compile(optimizer=optimizer, loss=nll_criterion)
    model.fit(dataset, epochs=250, callbacks=[early_stop])

    log.info(f'Trained model temperature to {temp_scale.temperature}.')

    predictions = tf.keras.activations.softmax(x)
    model_to_export = tf.keras.models.Model(input_, predictions)
    model_to_export.trainable = True

    if not path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model_to_export.save(save_dir)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-d', '--dataset', help='Path to dataset .csv',
                        required=True)
    parser.add_argument('-s', '--save-dir', type=str, dest='save_dir',
                        default='models',
                        help="Where to save the models.")
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str,
                        help='Where to find exported model.')

    args, unparsed = parser.parse_known_args(argv)

    if len(unparsed) != 0:
        raise SystemExit("Unknown argument: {}".format(unparsed))

    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    conf = Config(args.config.replace('/', '.'))

    if args.dataset is not None:
        conf.dataset = args.dataset
    else:
        if conf.dataset is None:
            raise ValueError('Must supply either config.dataset or '
                             'dataset command line argument.')

    setup_logger()
    temp_scale(conf, args.save_dir, args.model_dir)


if __name__ == '__main__':
    main()
