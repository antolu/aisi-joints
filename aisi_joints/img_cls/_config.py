import datetime
import logging
from typing import Union, List, Dict

import yaml

log = logging.getLogger(__name__)


def if_in_else(data: dict, key: str, default_value: ...):
    if key in data:
        return data[key]
    else:
        log.debug(f'Could not find key {key}, replacing with default '
                  f'value: {default_value}')
        return default_value


class FitConfig:
    learning_rate: float
    lr: float

    epochs: int
    decay_steps: int

    def __init__(self, data: dict):
        self.learning_rate = data['learning_rate']
        self.lr = self.learning_rate

        self.epochs = data['epochs']

        self.decay_steps = if_in_else(data, 'decay_steps', 0)


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
        with open(config_path, 'r') as f:
            data = yaml.full_load(f)

        self.batch_size = if_in_else(data, 'batch_size', 32)
        self.bs = self.batch_size

        self.workers = if_in_else(data, 'workers', 4)

        self.base_model = data['base_model']

        self.train_data = data['train_data']
        self.validation_data = data['validation_data']

        self.transfer_config = FitConfig(data['transfer'])
        self.finetune_config = FitConfig(data['finetune'])

        self.class_weights = if_in_else(data, 'class_weights', {})

        self.log_dir = 'logs'
        self.checkpoint_dir = 'checkpoints'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
