import os
from argparse import ArgumentParser, Namespace
from importlib import import_module

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from self_supervised import LinearClassifierMethod
from self_supervised.moco import SelfSupervisedMethod
from self_supervised.model_params import VICRegParams


def train(dataset_path: str, checkpoint_dir: str, config):
    os.environ['DATA_PATH'] = dataset_path
    params = config.model_params
    classifier_params = config.classifier_params

    model = SelfSupervisedMethod(params)

    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir,
        'model-{epoch:02d}-{loss:.2f}',
        monitor='loss',
        save_top_k=5,
        auto_insert_metric_name=False)

    trainer = pl.Trainer(accelerator='auto',
                         callbacks=[checkpoint_callback],
                         max_epochs=params.max_epochs)
    trainer.fit(model)

    linear_model = LinearClassifierMethod(classifier_params)
    linear_model.load_state_dict({k: v for k, v in model.state_dict().items()
                                  if k.startswith('model.')}, strict=False)
    trainer = pl.Trainer(accelerator='auto')

    trainer.fit(linear_model)


def main(args: Namespace, config):
    train(args.dataset, args.checkpoint_dir, config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-d', '--dataset', help='Path to dataset .csv',
                        required=True)
    parser.add_argument('-c', '--checkpoint-dir', dest='checkpoint_dir',
                        help='Path to checkpoint dir.', default='checkpoints')

    args = parser.parse_args()

    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    config_module = import_module(args.config.replace('/', '.'))

    main(args, config_module)
