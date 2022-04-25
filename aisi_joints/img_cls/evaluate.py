import logging
from argparse import ArgumentParser, Namespace
from pprint import pformat

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from ._dataloader import load_df
from ..constants import CLASS_OK, CLASS_DEFECT
from ..utils.logging import setup_logger
from ..utils.utils import time_execution

log = logging.getLogger(__name__)


def evaluate(df: pd.DataFrame, model: tf.keras.models.Model) -> pd.DataFrame:
    """
    Calculate predictions based on raw data and run COCO eval on it.
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


def main(args: Namespace):
    df = pd.read_csv(args.input)

    if args.split is not None and 'split' in df.columns:
        df = df[df['split'] == args.split]
        log.info(f'Evaluating on {args.split} split with {len(df)} samples.')
    else:
        log.info('Evaluating on entire dataset.')

    log.info(f'Loading model from {args.model_dir}')
    model = tf.keras.models.load_model(args.model_dir)
    log.info('Model loaded.')

    df = evaluate(df, model)

    if args.output is not None:
        log.info(f'Writing predictions to {args.output}')
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-i', '--input', required=True,
                        help='Path to .csv containing dataset, '
                             'or path to directory containing images.')
    parser.add_argument('-m', '--model-dir', dest='model_dir',
                        default='export_models',
                        help='Path to directory containing exported model.')
    parser.add_argument('-s', '--split',
                        choices=['train', 'validation', 'test'],
                        default=None,
                        help='Specific split to evaluate on.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .csv with predictions.')

    args = parser.parse_args()

    setup_logger()
    main(args)