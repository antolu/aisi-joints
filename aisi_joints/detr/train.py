"""
This module provides the script for training DE:TR on a COCO formatted dataset.

Use the `aisi_joints.data.coco_format` module to convert a .csv dataset to
COCO format.

This module is runnable. Use the `-h` option to view usage.
"""
from argparse import ArgumentParser
from functools import partial
from typing import List, Optional

import torch
import yaml
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ._data import CocoDetection
from ._data import collate_fn
from ._detr import Detr
from .evaluate import detect
from ..tfod.evaluate import evaluate_and_print

torch.multiprocessing.set_sharing_strategy('file_system')


class EvalCallback(Callback):
    """
    Callback that performs detection and calculates COCO metrics
    for a given dataset (usually validation) on the end of the
    validation step, and prints the results to console.
    """
    def __init__(self, model: Detr, dataset: CocoDetection,
                 evaluate_every: int = 10):
        self._model = model
        self._dataset = dataset

        self._evaluate_every = evaluate_every

    def on_validation_end(self, trainer: Trainer, pl_module):
        if trainer.current_epoch % self._evaluate_every == 0:
            detected = detect(self._model, self._dataset, 0.0)

            evaluate_and_print(self._dataset.coco, detected)


def train(config: dict, img_folder: str):
    model = Detr(lr=config['lr'], lr_backbone=config['lr_backbone'],
                 weight_decay=config['weight_decay'],
                 momentum=config['momentum'], num_classes=2)
    feature_extractor = model.feature_extractor

    train_dataset = CocoDetection(img_folder, 'train',
                                  feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder, 'validation',
                                feature_extractor=feature_extractor)

    train_dataloader = DataLoader(
        train_dataset, collate_fn=partial(collate_fn, feature_extractor),
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['workers'])
    val_dataloader = DataLoader(
        val_dataset, collate_fn=partial(collate_fn, feature_extractor),
        batch_size=1, shuffle=False, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        'checkpoints',
        'model-{epoch:02d}-{validation_loss:.2f}',
        monitor='validation_loss',
        save_top_k=5)

    trainer = Trainer(max_epochs=config['epochs'], gradient_clip_val=1.0,
                      callbacks=[checkpoint_callback, EvalCallback(model, val_dataset)],
                      accelerator='auto')

    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        pass
    finally:
        detected = detect(model, val_dataset, 0.0)

        evaluate_and_print(val_dataset.coco, detected)


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()

    parser.add_argument('-d', '--data', dest='data_dir',
                        help='Path to COCO dataset folder.')
    parser.add_argument('config',
                        help='Path to config.yml')

    args = parser.parse_args(argv)

    with open(args.config) as f:
        conf = yaml.full_load(f)

    train(conf, args.data_dir)


if __name__ == '__main__':
    main()
