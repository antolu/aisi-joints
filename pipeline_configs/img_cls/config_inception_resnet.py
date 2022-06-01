import tensorflow as tf
import tensorflow_addons as tfa

base_model: str = 'inception_resnet_v2'
workers: int = 4
batch_size: int = 32

transfer_config: dict = {
    'optimizer': tfa.optimizers.AdamW(
        weight_decay=1e-2,
        learning_rate=1e-3,
    ),
    'epochs': 50
}

finetune_config: dict = {
    'optimizer': tfa.optimizers.AdamW(
        weight_decay=1e-2,
        learning_rate=1e-4,),
    'epochs': 1000
}

class_weights: dict = {
    0: 0.86,
    1: 1.2
}
