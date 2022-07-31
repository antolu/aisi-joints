import datetime
import logging
from importlib import import_module
from typing import Union, List, Dict

import tensorflow as tf

from .config import layers_to_freeze

log = logging.getLogger(__name__)


def if_in_else(data: dict, key: str, default_value: ... = None):
    if key in data:
        return data[key]
    else:
        if default_value is None:
            msg = f'Could not find required key {key}.'
            raise AttributeError(msg)
        else:
            log.debug(f'Could not find key {key}, replacing with default '
                      f'value: {default_value}')
            return default_value


def if_hasattr_else(data: ..., key: str, default_value: ... = None):
    if hasattr(data, key):
        return getattr(data, key)
    else:
        if default_value is None:
            msg = f'Could not find required attribute {key}.'
            raise AttributeError(msg)
        else:
            log.debug(
                f'Could not find attribute {key}, replacing with default '
                f'value: {default_value}')
            return default_value


class FitConfig:
    optimizer: tf.keras.optimizers.Optimizer
    epochs: int
    callbacks: List[tf.keras.callbacks.Callback]

    def __init__(self, data: dict):
        self.optimizer = if_in_else(data, 'optimizer')
        self.epochs = if_in_else(data, 'epochs', 1000)

        self.callbacks = if_hasattr_else(data, 'callbacks')


class Config:
    batch_size: int
    bs: int

    fc_hidden_dim: int
    fc_dropout: float
    fc_num_layers: int
    fc_activation: str

    workers: int

    base_model: str
    layers_to_freeze: List[int]

    transfer_config: FitConfig
    finetune_config: FitConfig

    class_weights: Dict[int, float]

    dataset: str
    log_dir: str
    checkpoint_dir: str
    timestamp: str

    def __init__(self, config_path: str):
        module = import_module(config_path)

        self.batch_size = if_hasattr_else(module, 'batch_size', 32)
        self.bs = self.batch_size

        self.fc_hidden_dim = if_hasattr_else(module, 'fc_hidden_dim', 2048)
        self.fc_dropout = if_hasattr_else(module, 'fc_dropout', 0.8)
        self.fc_num_layers = if_hasattr_else(module, 'fc_num_layers', 1)
        self.fc_activation = if_hasattr_else(module, 'fc_activation', 'relu')

        self.workers = if_hasattr_else(module, 'workers', 4)

        self.base_model = if_hasattr_else(module, 'base_model',
                                          'inception_resnet_v2')
        self.layers_to_freeze = if_hasattr_else(module, 'layers_to_freeze',
                                                'none')

        self.transfer_config = FitConfig(
            if_hasattr_else(module, 'transfer_config'))
        self.finetune_config = FitConfig(
            if_hasattr_else(module, 'finetune_config'))

        self.class_weights = if_hasattr_else(module, 'class_weights', {})

        self.log_dir = 'logs'
        self.checkpoint_dir = 'checkpoints'
        self.dataset = None
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
