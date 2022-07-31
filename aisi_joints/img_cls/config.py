from typing import List

import tensorflow as tf

base_model: str = 'inception_resnet_v2'
workers: int = 4
batch_size: int = 32

fc_hidden_dim: int = 2048
fc_dropout: float = 0.8
fc_num_layers: int = 1
fc_activation: str = 'relu'

layers_to_freeze: List[int] = list(range(618))

train_data: str = 'dataset/train.tfrecord'
validation_data: str = 'dataset/validation.tfrecord'

transfer_config: dict = {
    'optimizer': tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            25, 0.94, staircase=True
        )
    ),
    'epochs': 1000,
    'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=5,
                                                   restore_best_weights=True)]
}

finetune_config: dict = {
    'optimizer': tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            0.0001,
            decay_steps=20000, alpha=1e-6
        )),
    'epochs': 10000,
    'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=5,
                                                   restore_best_weights=True)]
}

class_weights: dict = {
    0: 0.6,
    1: 2.2
}
