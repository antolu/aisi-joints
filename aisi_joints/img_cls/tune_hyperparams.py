import logging
import os
from argparse import ArgumentParser
from functools import partial
from os import path
from typing import List

import tensorflow as tf
from keras import Model
from keras.losses import CategoricalCrossentropy
from keras.metrics import Metric
from keras_tuner import HyperParameters, Hyperband, Objective
from tensorflow_addons.optimizers import AdamW

from ._config import Config
from ._dataloader import JointsSequence
from ._models import get_model
from .train import TensorBoardTool
from .._utils.logging import setup_logger
from .._utils.utils import get_latest

log = logging.getLogger(__name__)


def model_builder_full(hp: HyperParameters, config: Config,
                       train_base_model: bool = False,
                       metrics: List[Metric] = None) -> Model:
    """
    Build model for hyperparameters tuning

    hp: HyperParameters class instance
    """
    if metrics is None:
        metrics = []

    fc_hidden_dim = hp.Choice('fc_hidden_dim', [1024, 1536, 2048, 3072, 4096])
    # fc_num_layers = hp.Choice('fc_num_layers', [1, 2, 3])
    fc_dropout = hp.Float('fc_dropout', 0.3, 1.0)

    base_lr = hp.Float('lr', 1.e-5, 1.e-1, sampling='log')
    weight_decay = hp.Float('weight_decay', 1.e-5, 1.e-1, sampling='log')

    base_model, model, _ = get_model(config.base_model, fc_hidden_dim,
                                     fc_dropout)

    if not train_base_model:
        base_model.trainable = False

    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, decay_steps=50, decay_rate=0.94
    )
    optimizer = AdamW(weight_decay, lr_scheduler)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
                  metrics=metrics)

    return model


def model_builder_optimizer(hp: HyperParameters, model: Model,
                            metrics: List[Metric] = None) -> Model:
    """
    Build model for hyperparameters tuning

    hp: HyperParameters class instance
    """
    if metrics is None:
        metrics = []

    base_lr = hp.Float('lr', 1.e-5, 1.e-1, sampling='log')
    weight_decay = hp.Float('weight_decay', 1.e-5, 1.e-1, sampling='log')

    model.trainable = True

    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        base_lr, decay_steps=30000, alpha=1.e-6
    )
    optimizer = AdamW(weight_decay, lr_scheduler)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
                  metrics=metrics)

    return model


def main(dataset_csv: str, config: Config, mode: str = 'both'):
    base_model, model, _ = get_model(config.base_model, config.fc_hidden_dim,
                                     config.fc_dropout)
    input_size = base_model.input_shape[1:3]

    train_data = JointsSequence(dataset_csv, 'train', *input_size,
                                batch_size=config.batch_size)
    val_data = JointsSequence(dataset_csv, 'validation', *input_size,
                              random_crop=False, augment_data=False,
                              batch_size=config.batch_size)

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(class_id=0, name='precision_OK'),
               tf.keras.metrics.Precision(class_id=1, name='precision_DEFECT'),
               tf.keras.metrics.Recall(class_id=0, name='recall_OK'),
               tf.keras.metrics.Recall(class_id=1, name='recall_DEFECT')]

    separator = 80 * '='

    # =========================================================================
    # Transfer
    # =========================================================================
    if mode in ('both', 'transfer'):
        log.info('\n'.join([separator,
                            'Tuning fully connected layers',
                            separator]))

        tuner = Hyperband(hypermodel=partial(model_builder_full, config=config,
                                             train_base_model=False,
                                             metrics=metrics),
                          objective=Objective('val_accuracy', direction='max'),
                          max_epochs=50, project_name='Transfer')

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      patience=5)

        tuner.search(train_data,
                     validation_data=val_data,
                     batch_size=config.batch_size,
                     epochs=50,
                     shuffle=True,
                     initial_epoch=0,
                     use_multiprocessing=False,
                     workers=config.workers,
                     class_weight=config.class_weights,
                     callbacks=[stop_early])

        log.info(separator)
        log.info('Transfer tuning results')
        log.info(tuner.results_summary())
        log.info(separator)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_transfer = best_hps

        model = tuner.hypermodel.build(best_hps)

        # =====================================================================
        # Train fully connected layers
        # =====================================================================
        log.info('\n'.join([separator,
                            'Training fully connected layers',
                            separator]))

        if not path.isdir(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=path.join(config.checkpoint_dir,
                               'model.{epoch:02d}-{val_loss:.2f}.h5'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=path.join(config.log_dir, config.timestamp, 'transfer'))

        # train the model on the new data for a few epochs
        model.fit(train_data, batch_size=config.batch_size,
                  validation_data=val_data, use_multiprocessing=False,
                  workers=config.workers, epochs=config.transfer_config.epochs,
                  class_weight=config.class_weights,
                  callbacks=[model_checkpoint_callback, tb_callback])

    # =========================================================================
    # Fine tuning
    # =========================================================================
    if mode in ('both', 'finetune'):
        model.load_weights(get_latest(config.checkpoint_dir,
                                      lambda o: o.endswith('.h5')))

        log.info('\n'.join([separator,
                            'Tuning optimizer for fine tuning',
                            separator]))

        tuner = Hyperband(hypermodel=partial(model_builder_optimizer,
                                             model=model,
                                             metrics=metrics),
                          objective=Objective('val_accuracy', direction='max'),
                          max_epochs=50, project_name='Finetune')

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      patience=5)

        tuner.search(train_data,
                     validation_data=val_data,
                     batch_size=config.batch_size,
                     epochs=50,
                     shuffle=True,
                     initial_epoch=0,
                     use_multiprocessing=False,
                     workers=config.workers,
                     class_weight=config.class_weights,
                     callbacks=[stop_early])

        # =====================================================================
        # Train full model
        # =====================================================================
        log.info(separator)
        log.info('Finetune results')
        log.info(tuner.results_summary())
        log.info(separator)

        best_hps_finetune = tuner.get_best_hyperparameters(num_trials=1)[0]

        log.info('\n'.join([separator, 'Training full model', separator]))

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=path.join(config.checkpoint_dir,
                               'model.{epoch:02d}-{val_loss:.2f}.h5'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=path.join(config.log_dir, config.timestamp, 'finetune'))

        model = tuner.hypermodel.build(best_hps_finetune)

        model.fit(train_data, batch_size=config.batch_size,
                  validation_data=val_data, use_multiprocessing=False,
                  workers=config.workers, epochs=config.transfer_config.epochs,
                  class_weight=config.class_weights,
                  callbacks=[model_checkpoint_callback, tb_callback])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')
    parser.add_argument('-d', '--dataset', help='Path to dataset .csv',
                        required=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Launch tensorboard as part of the script.')
    parser.add_argument('-c', '--checkpoint-dir', default='checkpoints',
                        dest='checkpoint_dir',
                        help='Directory to save checkpoint files.')
    parser.add_argument('-m', '--mode',
                        choices=['both', 'transfer', 'finetune'],
                        default='both',
                        help='Perform full transfer learning, or just the'
                             'fully connected layers, or finetuning the cnn.')

    args = parser.parse_args()

    setup_logger()
    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    config = Config(args.config.replace('/', '.'))
    config.log_dir = args.logdir
    config.checkpoint_dir = args.checkpoint_dir

    if args.tensorboard:
        tensorboard = TensorBoardTool(config.log_dir)
        tensorboard.run()

    try:
        main(args.dataset, config, args.mode)
    except KeyboardInterrupt:
        pass
