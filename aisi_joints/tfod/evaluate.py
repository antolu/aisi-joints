""""
This script serves as a helper to evaluate an exported tensorflow object detection
model on a custom formatted dataset (used in this package).
"""
import logging
from argparse import ArgumentParser, Namespace

import pandas as pd
import tensorflow as tf
from object_detection.metrics.coco_tools import COCOWrapper, COCOEvalWrapper
from pycocotools.coco import COCO

from .detect import dataframe_detect
from ..constants import LABEL_MAP
from ..data.coco_format import df_to_coco

log = logging.getLogger(__name__)


def format_metrics(metrics: dict):
    """
    Convert COCO tfod metrics to printable message.
    """
    max_len = max([len(o) for o in list(metrics.keys())])

    msg = ''
    for metric, value in metrics.items():
        msg += (metric + ' ' * (max_len - len(metric)))
        msg += ' = '
        msg += f'{value:.3f}\n'

    return msg


def evaluate_and_print(coco_gt: COCO, coco_pred: COCO):
    coco_eval = COCOEvalWrapper(coco_gt, coco_pred)

    metrics, per_cat_metrics = coco_eval.ComputeMetrics(
        include_metrics_per_category=True,
        all_metrics_per_category=True)

    print('=' * 79)
    print(format_metrics(metrics))
    print('=' * 79)
    print(format_metrics(per_cat_metrics))


def evaluate(df: pd.DataFrame, model: tf.keras.Model,
             score_threshold: float = 0.5):
    """
    Calculate predictions based on raw data and run COCO tfod on it. Print results
    to terminal.
    """
    inv_label_map = {idx: {'name': name, 'id': idx} for name, idx in
                     LABEL_MAP.items()}

    predictions = dataframe_detect(df, model, inv_label_map, score_threshold)

    coco_groundtruth = df_to_coco(df, LABEL_MAP)
    coco_predictions = df_to_coco(predictions, LABEL_MAP, predictions=True)

    coco_gt = COCOWrapper(coco_groundtruth)
    coco_pred = COCOWrapper(coco_predictions)

    coco_eval = COCOEvalWrapper(coco_gt, coco_pred)

    evaluate_and_print(coco_gt, coco_pred)


def main(args: Namespace):
    df = pd.read_csv(args.input)

    model = tf.saved_model.load(args.model_dir)

    if args.split is not None and 'split' in df.columns:
        df = df[df['split'] == args.split]

    evaluate(df, model, args.score_threshold)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-i', '--input', required=True,
                        help='Path to .csv containing dataset, '
                             'or path to directory containing images.')
    parser.add_argument('-l', '--labelmap', default=None,
                        help='Path to label map file.')
    parser.add_argument('-m', '--model-dir', dest='model_dir', required=True,
                        help='Path to directory containing exported model.')
    parser.add_argument('-s', '--split',
                        choices=['train', 'validation', 'test'],
                        default=None,
                        help='Specific split to evaluate on.')
    parser.add_argument('-t', '--score-threshold', dest='score_threshold',
                        default=0.5,
                        help='Detection score threshold. All detections under '
                             'this confidence score are discarded.')

    args = parser.parse_args()

    main(args)
