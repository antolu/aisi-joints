import tensorflow as tf

base_model: str = 'inception_resnet_v2'
workers: int = 8
batch_size: int = 32

fc_hidden_dim: int = 1024
fc_dropout: float = 0.8
fc_num_layers: int = 1
fc_activation = 'relu'

layers_to_freeze = 'none'

transfer_config: dict = {
    'optimizer': tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.005, 25, 0.94),
    ),
    'epochs': 50,
    'callbacks': [tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True)]
}

finetune_config: dict = {
    'optimizer': tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(0.00004,
                                                                        3000, alpha=1.e-8)
    ),
    'epochs': 50
}

class_weights: dict = {
    0: 0.9,
    1: 1.16
}
