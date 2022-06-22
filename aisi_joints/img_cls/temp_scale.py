import argparse
import logging
import os
from os import path
from typing import Iterable, Optional

import tensorflow as tf

from ._config import Config
from ._dataloader import JointsSequence
from .._utils import setup_logger

log = logging.getLogger(__name__)


class TempScale(tf.keras.layers.Layer):
    def __init__(self, temperature: Optional[float] = None):
        super().__init__()

        if temperature is None:
            temperature = 1.5
        self._temperature = tf.Variable(temperature, dtype="float32",
                                        trainable=True)

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


@tf.function
def get_logits(model: tf.keras.Model, dataset: Iterable):
    all_logits = []
    all_labels = []

    for images, labels in dataset:
        logits = model(images)

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = tf.concat(all_logits, 0, name='logits_valid')
    all_labels = tf.concat(all_labels, 0, name='labels_valid')

    return all_logits, all_labels


def main(config: Config, save_dir: str, model_dir: str):
    model: tf.keras.Model = tf.keras.models.load_model(model_dir)
    input_size = model.input_shape[1:3]

    dataset = JointsSequence(config.dataset, 'validation', *input_size,
                             random_crop=False, augment_data=False,
                             batch_size=config.batch_size)

    # remove softmax layer
    input_ = model.input
    classification_layer = model.layers[-1]
    classification_layer.activation = tf.keras.activations.linear
    model.trainable = False  # freeze all layers

    model_wo_softmax = model

    logits, labels = get_logits(model_wo_softmax, dataset)
    logits = tf.constant(logits)
    labels = tf.constant(labels)

    @tf.function
    def temp_scale(logits, temperature):
        return logits / temperature

    temp_var = tf.Variable(1., name='temperature', trainable=True)

    nll_loss = tf.losses.CategoricalCrossentropy(from_logits=True)

    def compute_loss() -> tf.Tensor:
        logits_w_temp = tf.divide(logits, temp_var)
        loss = nll_loss(labels, logits_w_temp)

        return loss

    optimizer = tf.keras.optimizers.SGD(learning_rate=1)

    tol = 1e-3
    old_val = 1e8
    for i in range(1000):
        optimizer.minimize(compute_loss, [temp_var])

        loss = compute_loss()
        if tf.math.abs(loss - old_val) < tol:
            break
        else:
            old_val = loss

    log.info(f'Trained model temperature to {temp_scale.temperature} after '
             f'{i} iterations.')

    predictions = tf.keras.activations.softmax(
        tf.divide(classification_layer.output, temp_var))
    model_to_export = tf.keras.models.Model(input_, predictions)
    model_to_export.trainable = True

    if not path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model_to_export.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-d', '--dataset', help='Path to dataset .csv',
                        required=True)
    parser.add_argument('--save-dir', type=str, dest='save_dir',
                        default='models',
                        help="Where to save the models.")
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        help='Where to find exported model.')

    args, unparsed = parser.parse_known_args()

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
    main(conf, args.save_dir, args.model_dir)
