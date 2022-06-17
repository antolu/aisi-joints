import tensorflow as tf
import tensorflow_addons as tfa


base_model: str = 'inception_resnet_v2'
workers: int = 4
batch_size: int = 32

fc_hidden_dim: int = 4096
fc_dropout: float = 0.8
fc_num_layers: int = 2

transfer_config: dict = {
    'optimizer': tf.keras.optimizers.SGD(
        momentum=0.95,
        learning_rate=0.005,
    ),
    'epochs': 50
}

finetune_config: dict = {
    'optimizer': tf.keras.optimizers.SGD(
        momentum=0.95,
        learning_rate=0.05,),
    'epochs': 1000
}

class_weights: dict = {
    0: 0.86,
    1: 1.2
}
