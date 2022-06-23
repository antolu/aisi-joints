import logging
from argparse import ArgumentParser, Namespace
from pprint import pformat
from typing import List, Optional

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from ._dataloader import load_df
from ..constants import CLASS_OK, CLASS_DEFECT
from .._utils.logging import setup_logger
from .._utils import time_execution, get_latest
from ._config import Config
from ._models import get_model

log = logging.getLogger(__name__)


def evaluate(df: pd.DataFrame, model: tf.keras.models.Model) -> pd.DataFrame:
    """
    Calculate predictions based on raw data and run COCO tfod on it.
    Print results to terminal.
    """
    dataset = load_df(df, random_crop=False, augment_data=False)
    class_map = {CLASS_OK: 0, CLASS_DEFECT: 1}
    labels = df['cls'].map(class_map).to_numpy()
    total = len(labels)

    log.info('Running inference...')
    dataset = dataset.batch(32)

    with time_execution() as t:
        predictions = model.predict(dataset, batch_size=32)

    log.info(f'Done. Took {t.duration * 1000 / total} ms per sample.')
    log.info('Calculating evaluation metrics.')

    pred_labels = tf.argmax(predictions, axis=1).numpy()
    scores = tf.reduce_max(predictions, axis=1).numpy()

    report = classification_report(
        labels, pred_labels, target_names=[CLASS_OK, CLASS_DEFECT],
        output_dict=True)
    cf = confusion_matrix(labels, pred_labels)

    log.info(('=' * 10) + 'CLASSIFICATION REPORT' + ('=' * 10)
             + '\n' + pformat(report) + '\n')
    log.info(('=' * 10) + 'CONFUSION MATRIX' + ('=' * 10) + '\n' + pformat(cf))

    df = df.assign(detected_class=pred_labels)
    df['detected_class'] = df['detected_class'].map(
        {v: k for k, v in class_map.items()})
    df['detection_score'] = scores
    df['num_detections'] = 1
    df['detected_x0'] = df['x0']
    df['detected_x1'] = df['x1']
    df['detected_y0'] = df['y0']
    df['detected_y1'] = df['y1']

    return df


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset', required=True,
                        help='Path to .csv containing dataset, '
                             'or path to directory containing images.')
    parser.add_argument('-m', '--model-dir', dest='model_dir',
                        default='export_models',
                        help='Path to directory containing exported model '
                             'or checkpoint.')
    parser.add_argument('-c', '--config', default=None,
                        help='Path to config.py if evaluating checkpoint.')
    parser.add_argument('-s', '--split',
                        choices=['train', 'validation', 'test'],
                        default=None,
                        help='Specific split to evaluate on.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .csv with predictions.')

    args = parser.parse_args(argv)

    setup_logger()
    df = pd.read_csv(args.dataset)

    if args.split is not None and 'split' in df.columns:
        df = df[df['split'] == args.split]
        log.info(f'Evaluating on {args.split} split with {len(df)} samples.')
    else:
        log.info('Evaluating on entire dataset.')

    if args.config is None:
        log.info(f'Loading model from {args.model_dir}')
        model = tf.keras.models.load_model(args.model_dir)
    else:
        if args.config.endswith('.py'):
            args.config = args.config[:-3]
        config = Config(args.config.replace('/', '.'))
        base_model, model, _ = get_model(config.base_model,
                                         config.fc_hidden_dim,
                                         config.fc_dropout,
                                         config.fc_num_layers,
                                         config.fc_activation)

        checkpoint_path = get_latest(args.model_dir,
                                     lambda o: o.endswith('.h5'))

        log.info(f'Loading model weights from {checkpoint_path}.')
        model.load_weights(checkpoint_path)

    log.info('Model loaded.')

    df = evaluate(df, model)

    if args.output is not None:
        log.info(f'Writing predictions to {args.output}')
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
