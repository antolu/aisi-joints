import logging
from typing import Tuple, Callable

import tensorflow as tf
from keras import Model, Input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

log = logging.getLogger(__name__)


def get_model(model_name: str, fc_hidden_dim: int = 2048,
              fc_dropout: float = 0.8) \
        -> Tuple[Model, Model, Callable]:
    if model_name == 'inception_resnet_v2':
        base_model: Model = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights='imagenet',
            input_tensor=Input(shape=(299, 299, 3)))
        preprocess_fn = tf.keras.applications.inception_resnet_v2 \
            .preprocess_input
    elif model_name == 'vgg19':
        base_model: Model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet',
            input_tensor=Input(shape=(299, 299, 32)))
        preprocess_fn = tf.keras.applications.vgg19.preprocess_input
    elif model_name == 'resnet101v2':
        base_model: Model = tf.keras.applications.ResNet101V2(
            include_top=False, weights='imagenet',
            input_tensor=Input(shape=(299, 299, 32)))
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'resnet152v2':
        base_model: Model = tf.keras.applications.ResNet152V2(
            include_top=False, weights='imagenet',
            input_tensor=Input(shape=(299, 299, 32)))
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'efficientnetv2l':
        base_model: Model = tf.keras.applications.EfficientNetV2L(
            include_top=False, weights='imagenet',
            input_tensor=Input(shape=(299, 299, 3)))
        preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        raise NotImplementedError

    input_ = base_model.input
    preprocessed_input = preprocess_fn(input_)
    base_model = Model(inputs=input_,
                       outputs=base_model(preprocessed_input))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add fully connected layer
    x = Dense(fc_hidden_dim, activation='relu')(x)
    x = Dropout(1.0 - fc_dropout)(x)
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model, preprocess_fn
