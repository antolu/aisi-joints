from importlib import import_module
import datetime
import logging
from typing import Union, List, Dict
import tensorflow as tf

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
            log.debug(f'Could not find attribute {key}, replacing with default '
                      f'value: {default_value}')
            return default_value


class FitConfig:
    optimizer: tf.keras.optimizers.Optimizer
    epochs: int

    def __init__(self, data: dict):
        self.optimizer = if_in_else(data, 'optimizer')
        self.epochs = if_in_else(data, 'epochs', 1000)


class Config:
    batch_size: int
    bs: int

    workers: int

    base_model: str

    train_data: Union[str, List[str]]
    validation_data: Union[str, List[str]]

    transfer_config: FitConfig
    finetune_config: FitConfig

    class_weights: Dict[int, float]

    log_dir: str
    checkpoint_dir: str
    timestamp: str

    def __init__(self, config_path: str):
        module = import_module(config_path)

        self.batch_size = if_hasattr_else(module, 'batch_size', 32)
        self.bs = self.batch_size

        self.workers = if_hasattr_else(module, 'workers', 4)

        self.base_model = if_hasattr_else(module, 'base_model',
                                          'inception_resnet_v2')

        self.train_data = if_hasattr_else(module, 'train_data')
        self.validation_data = if_hasattr_else(module, 'validation_data')

        self.transfer_config = FitConfig(
            if_hasattr_else(module, 'transfer_config'))
        self.finetune_config = FitConfig(
            if_hasattr_else(module, 'finetune_config'))

        self.class_weights = if_hasattr_else(module, 'class_weights', {})

        self.log_dir = 'logs'
        self.checkpoint_dir = 'checkpoints'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
