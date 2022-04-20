import logging
from typing import Tuple

import tensorflow as tf
from keras import Model, Input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

log = logging.getLogger(__name__)


def get_model(model_name: str) -> Tuple[Model, Model]:
    if model_name == 'inception_resnet_v2':
        base_model: Model = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights='imagenet', input_tensor=Input(shape=(299, 299, 3)))
    else:
        raise NotImplementedError

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add fully connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model
