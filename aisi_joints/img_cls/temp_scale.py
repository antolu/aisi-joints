import argparse
import os
from os import path

import tensorflow as tf

from ._config import Config
from ._dataloader import load_tfrecord


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


def main(config: Config, save_dir: str, model_dir: str):
    dataset = load_tfrecord(config.validation_data, config.batch_size,
                            shuffle=False, random_crop=False,
                            augment_data=False)

    model: tf.keras.models.Model = tf.keras.models.load_model(model_dir)

    input_ = model.input
    classification_layer = model.layers[-1]
    classification_layer.activation = tf.keras.activations.linear
    model.trainable = False  # freeze all layers

    temp_scale = TempScale()
    x = temp_scale(classification_layer.output)
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

    print(f'Trained model temperature to {temp_scale.temperature}.')

    predictions = tf.keras.activations.softmax(x)
    model_to_export = tf.keras.models.Model(input_, predictions)
    model_to_export.trainable = True

    if not path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model_to_export.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Path to config.py')
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

    main(conf, args.save_dir, args.model_dir)
