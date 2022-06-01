import tensorflow as tf
import tensorflow_addons as tfa

base_model: str = 'efficientnetv2l'
workers: int = 4
batch_size: int = 16

fc_hidden_dim: int = 1792
fc_dropout: float = 0.8

transfer_config: dict = {
    'optimizer': tfa.optimizers.AdamW(
        weight_decay=1e-4,
        learning_rate=1e-3,
    ),
    'epochs': 50
}

finetune_config: dict = {
    'optimizer': tfa.optimizers.AdamW(
        weight_decay=1e-5,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 500, 0.97),),
    'epochs': 500
}

class_weights: dict = {
    0: 0.86,
    1: 1.2
}
