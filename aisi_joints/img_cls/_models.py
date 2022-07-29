import logging
from typing import List, Optional, Union

import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense, Dropout
from keras import activations

from ._config import Config

log = logging.getLogger(__name__)


class MLP(tf.keras.layers.Layer):
    """Sequential multi-layer perceptron (MLP) block."""

    def __init__(
            self,
            units: List[int],
            use_bias: bool = True,
            activation: Optional[str] = "relu",
            dropout: Optional[float] = 0.8,
            final_activation: Optional[str] = None,
            **kwargs) -> None:
        """Initializes the MLP layer.
        Args:
          units: Sequential list of layer sizes.
          use_bias: Whether to include a bias term.
          activation: Type of activation to use on all except the last layer.
          final_activation: Type of activation to use on last layer.
          **kwargs: Extra args passed to the Keras Layer base class.
        """

        super().__init__(**kwargs)

        self._sublayers = []
        self._units = units
        self._use_bias = use_bias
        self._activation = activation
        self._dropout = dropout
        self._final_activation = final_activation

        if len(units) == 0:
            raise ValueError('List of units must not be empty.')

        if len(units) > 1:
            for num_units in units[:-1]:
                self._sublayers.append(
                    Dense(
                        num_units, activation=activation, use_bias=use_bias))
                self._sublayers.append(Dropout(1.0 - dropout))
        self._sublayers.append(
            Dense(
                units[-1], activation=None, use_bias=use_bias))

        # separate final activation from FCs to simplify temp scaling
        self._sublayers.append(activations.get(final_activation))

    def get_config(self):
        config = super().get_config()

        config.update({
            'units': self._units,
            'use_bias': self._use_bias,
            'activation': self._activation,
            'dropout': self._dropout,
            'final_activation': self._final_activation,
        })

        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward computation of the block."""
        for layer in self._sublayers:
            x = layer(x)

        return x


class ModelWrapper:
    def __init__(self, config: Config):
        self._config = config

        self._model = None

        self.freeze_mode = 'base'

    @property
    def model(self) -> Model:
        c = self._config

        if self._model is None:
            self._model = get_model(c.base_model,
                                    c.fc_hidden_dim,
                                    c.fc_dropout,
                                    c.fc_num_layers,
                                    c.fc_activation)

        return self._model

    @property
    def config(self):
        return self._config

    def freeze(self):
        if self.freeze_mode == 'base':
            self.model.layers[-2].trainable = False
        elif self.freeze_mode == 'partial':
            freeze_layers(self.model.layers[-2], self._config.layers_to_freeze)


def get_model(model_name: str, fc_hidden_dim: int = 2048,
              fc_dropout: float = 0.8, fc_num_layers: int = 1,
              fc_activation: str = 'relu') -> Model:
    if model_name == 'inception_resnet_v2':
        base_model: Model = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights='imagenet', pooling='avg',
            input_tensor=Input(shape=(299, 299, 3)))
        preprocess_fn = tf.keras.applications.inception_resnet_v2 \
            .preprocess_input
    elif model_name == 'vgg19':
        base_model: Model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet', pooling='avg',
            input_tensor=Input(shape=(299, 299, 32)))
        preprocess_fn = tf.keras.applications.vgg19.preprocess_input
    elif model_name == 'resnet101v2':
        base_model: Model = tf.keras.applications.ResNet101V2(
            include_top=False, weights='imagenet', pooling='avg',
            input_tensor=Input(shape=(299, 299, 32)))
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'resnet152v2':
        base_model: Model = tf.keras.applications.ResNet152V2(
            include_top=False, weights='imagenet', pooling='avg',
            input_tensor=Input(shape=(299, 299, 32)))
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'efficientnetv2l':
        base_model: Model = tf.keras.applications.EfficientNetV2L(
            include_top=False, weights='imagenet', pooling='avg',
            input_tensor=Input(shape=(299, 299, 3)))
        preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        raise NotImplementedError

    input_ = base_model.input
    preprocessed_input = preprocess_fn(input_)
    base_model = Model(inputs=input_,  # add training=False for BN layers
                       outputs=base_model(preprocessed_input, training=False))

    # add fully connected layer
    mlp = MLP(([fc_hidden_dim] * fc_num_layers) + [2],
              activation=fc_activation,
              dropout=fc_dropout, final_activation='softmax')
    predictions = mlp(base_model.output)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def freeze_layers(model: Model,
                  layers_to_freeze: Union[List[int], str] = 'none'):
    if layers_to_freeze == 'all':
        model.trainable = False
    elif layers_to_freeze == 'none':
        pass
    else:
        for i in layers_to_freeze:
            model.layers[i].trainable = False
