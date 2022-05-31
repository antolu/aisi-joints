import logging
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

from .._utils.logging import setup_logger
from .._utils.utils import get_latest
from ._dataloader import prepare_dataset
from .train import TensorBoardTool
from ._config import Config
from ._models import get_model

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

    fc_hidden_dim = hp.Choice('fc_hidden_dim', [1024, 1536, 2048])
    fc_dropout = hp.Float('fc_dropout', 0.5, 1.0)

    base_lr = hp.Float('lr', 0.0001, 0.01, sampling='log')
    weight_decay = hp.Float('weight_decay', 1.e-5, 1., sampling='log')

    base_model, model, _ = get_model(config.base_model, fc_hidden_dim,
                                     fc_dropout)

    if not train_base_model:
        base_model.trainable = False

    optimizer = AdamW(weight_decay, base_lr)
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

    base_lr = hp.Float('lr', 0.0001, 0.01, sampling='log')
    weight_decay = hp.Float('weight_decay', 1.e-5, 1., sampling='log')

    model.trainable = True
    optimizer = AdamW(weight_decay, base_lr)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
                  metrics=metrics)

    return model


def main(config: Config):
    train_data = tf.data.TFRecordDataset(config.train_data)
    val_data = tf.data.TFRecordDataset(config.validation_data)

    base_model, model, _ = get_model(config.base_model, config.fc_hidden_dim,
                                     config.fc_dropout)
    input_size = base_model.input_shape[1:3]

    train_data = prepare_dataset(train_data, *input_size, config.bs,
                                 random_crop=True)
    val_data = prepare_dataset(val_data, *input_size, config.bs, shuffle=False,
                               random_crop=False, augment_data=False)

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(class_id=0, name='precision_OK'),
               tf.keras.metrics.Precision(class_id=1, name='precision_DEFECT'),
               tf.keras.metrics.Recall(class_id=0, name='recall_OK'),
               tf.keras.metrics.Recall(class_id=1, name='recall_DEFECT')]

    separator = 80 * '='
    print('\n'.join([separator, 'Tuning fully connected layers', separator]))

    tuner = Hyperband(hypermodel=partial(model_builder_full, config=config,
                                         train_base_model=False,
                                         metrics=metrics),
                      objective=Objective('accuracy', direction='max'),
                      max_epochs=50, project_name='Transfer')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5)

    tuner.search(training_data=train_data,
                 validation_data=val_data,
                 batch_size=config.batch_size,
                 epochs=50,
                 shuffle=True,
                 initial_epoch=0,
                 use_multiprocessing=True,
                 workers=config.workers,
                 class_weight=config.class_weights,
                 callbacks=[stop_early])

    print(separator)
    print('Transfer tuning results')
    print(tuner.results_summary())
    print(separator)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps_transfer = best_hps

    model = tuner.hypermodel.build(best_hps)

    print('\n'.join([separator, 'Training fully connected layers', separator]))

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
              validation_data=val_data, use_multiprocessing=True,
              workers=config.workers, epochs=config.transfer_config.epochs,
              class_weight=config.class_weights,
              callbacks=[model_checkpoint_callback, tb_callback])

    model.load_weights(get_latest(config.checkpoint_dir,
                                  lambda o: o.endswith('.h5')))

    print('\n'.join([separator,
                     'Tuning optimizer for fine tuning',
                     separator]))

    tuner = Hyperband(hypermodel=partial(model_builder_optimizer,
                                         model=model,
                                         metrics=metrics),
                      objective=Objective('accuracy', direction='max'),
                      max_epochs=50, project_name='Finetune')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                                  patience=5)

    tuner.search(training_data=train_data,
                 validation_data=val_data,
                 batch_size=config.batch_size,
                 epochs=50,
                 shuffle=True,
                 initial_epoch=0,
                 use_multiprocessing=True,
                 workers=config.workers,
                 class_weight=config.class_weights,
                 callbacks=[stop_early])

    print(separator)
    print('Finetune results')
    print(tuner.results_summary())
    print(separator)

    best_hps_finetune = tuner.get_best_hyperparameters(num_trials=1)[0]

    print('\n'.join([separator, 'Training full model', separator]))

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
              validation_data=val_data, use_multiprocessing=True,
              workers=config.workers, epochs=config.transfer_config.epochs,
              class_weight=config.class_weights,
              callbacks=[model_checkpoint_callback, tb_callback])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config.py')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Tensorboard logdir.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug logs')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Launch tensorboard as part of the script.')
    parser.add_argument('-c', '--checkpoint-dir', default='checkpoints',
                        dest='checkpoint_dir',
                        help='Directory to save checkpoint files.')

    args = parser.parse_args()

    setup_logger(args.debug)
    if args.config.endswith('.py'):
        args.config = args.config[:-3]
    config = Config(args.config.replace('/', '.'))
    config.log_dir = args.logdir
    config.checkpoint_dir = args.checkpoint_dir

    if args.tensorboard:
        tensorboard = TensorBoardTool(config.log_dir)
        tensorboard.run()

    try:
        main(config)
    except KeyboardInterrupt:
        pass
