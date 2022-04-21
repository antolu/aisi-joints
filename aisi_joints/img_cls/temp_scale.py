import argparse

import yaml
import tensorflow as tf
import tensorflow_probability as tfp

from .dataloader import load_tfrecord

from .config import Config


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


def main(config: Config, save_dir: str, model_dir: str):
    dataset = load_tfrecord(config.validation_data, config.batch_size,
                            shuffle=False, random_crop=False,
                            augment_data=False)

    model: tf.keras.models.Model = tf.keras.models.load_model(model_dir)
    model.trainable = False  # freeze all layers

    input_ = model.input
    classification_layer = model.layers[-1]
    classification_layer.activation = tf.keras.activations.linear

    temp_scale = TempScale()
    x = temp_scale(classification_layer.output)
    model = tf.keras.models.Model(input_, x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    nll_criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optmizer=optimizer)
    model.fit(dataset, batch_size=config.batch_size, epochs=10)

    predictions = tf.keras.activations.softmax(x)
    model_to_export = tf.keras.Models.Model(input_, predictions)
    model_to_export.trainable = True
    print(f'Trained model temperature to {temp_scale._temperature}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Path to config.yml')
    parser.add_argument('--save-dir', type=str, dest='save_dir',
                        default='models',
                        help="Where to save the models.")
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        help='Where to find exported model.')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) != 0:
        raise SystemExit("Unknown argument: {}".format(unparsed))

    conf = Config(args.config)

    main(conf, args.save_dir, args.model_dir)
