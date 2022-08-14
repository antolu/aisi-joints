"""
This module provides functions related to detection in an tfod exported model.

This module is runnable. Use the `-h` option to view usage.
"""
import logging
from argparse import ArgumentParser
from os import path
from typing import Dict, Union, Optional

import pandas as pd
import tensorflow as tf

from .utils import (
    load_model,
    load_labelmap,
    load_image,
    run_inference,
    format_detections,
)
from .._utils.logging import setup_logger
from .._utils.utils import time_execution

log = logging.getLogger(__name__)


def csv_detect(
    csv_path: str,
    model_path: str,
    label_map: Union[Dict[str, int], str],
    score_threshold: float = 0.5,
    output: Optional[str] = None,
):
    """
    Loads dataset from .csv file and model from exported model, and label map.

    Runs detection on samples in csv file. Writes results back to dataframe
    in columns 'detected_class', 'detection_score', 'detected_{x0,x1,y0,y1}',
    'num_detections'. Multiple detections are separated by a ';' character.

    Writes .csv with detections to output if specified.

    Parameters
    ----------
    csv_path: str
    model_path: str
    label_map: dict or str
    score_threshold: float
    output: str
    """
    df = pd.read_csv(csv_path)

    if isinstance(label_map, str) and path.isfile(label_map):
        label_map = load_labelmap(label_map)

    model = load_model(model_path)

    df = dataframe_detect(df, model, label_map, score_threshold)

    if output is not None:
        df.to_csv(output, index=False)
    else:
        df.to_csv(csv_path, index=False)


def dataframe_detect(
    df: pd.DataFrame,
    model: tf.keras.Model,
    label_map: Dict[int, dict],
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Runs detection on samples in passed dataframe. Writes results back to dataframe
    in columns 'detected_class', 'detection_score', 'detected_{x0,x1,y0,y1}',
    'num_detections'. Multiple detections are separated by a ';' character.

    Parameters
    ----------
    df: pd.DataFrame
    model: tf.keras.Model
    label_map: dict
    score_threshold: float

    Returns
    -------
    pd.DataFrame
    """
    results = []
    with time_execution() as t:
        for sample in df.itertuples():
            image = load_image(sample.filepath)
            detections = run_inference(image, model)

            boxes = format_detections(detections)

            boxes_filtered = boxes[
                boxes['detection_scores'] >= score_threshold
            ]

            detected_class = []
            x0 = []
            x1 = []
            y0 = []
            y1 = []
            detection_score = []

            for box in boxes_filtered.itertuples():
                detected_class.append(label_map[box.detection_classes]['name'])
                detection_score.append(box.detection_scores)
                x0.append(box.left)
                x1.append(box.right)
                y0.append(box.bottom)
                y1.append(box.top)

            res = {
                'eventId': sample.eventId,
                'detected_class': ';'.join(map(str, detected_class)),
                'detection_score': ';'.join(map(str, detection_score)),
                'detected_x0': ';'.join(map(str, map(int, x0))),
                'detected_x1': ';'.join(map(str, map(int, x1))),
                'detected_y0': ';'.join(map(str, map(int, y0))),
                'detected_y1': ';'.join(map(str, map(int, y1))),
                'num_detections': len(x0),
            }

            results.append(res)

    log.info(
        f'Finished detection, took {t.duration * 1000 / len(df)} ms '
        f'per sample.'
    )

    res_df = pd.DataFrame(results)

    df = df.copy()
    df = pd.merge(df, res_df, on='eventId')

    return df


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Path to .csv containing dataset, '
        'or path to directory containing images.',
    )
    parser.add_argument(
        '-l', '--labelmap', required=True, help='Path to label map file.'
    )
    parser.add_argument(
        '-m',
        '--model-dir',
        dest='model_dir',
        required=True,
        help='Path to directory containing exported model.',
    )
    parser.add_argument(
        '-t',
        '--score-threshold',
        dest='score_threshold',
        default=0.5,
        type=float,
        help='Detection score threshold. All detections under '
        'this confidence score are discarded.',
    )
    parser.add_argument(
        '-o', '--output', default='output', help='Output directory for images.'
    )
    parser.add_argument(
        '--save-plot',
        dest='save_plot',
        action='store_true',
        help='Save images with bounding boxes.',
    )
    args = parser.parse_args()

    setup_logger()
    csv_detect(
        args.input,
        args.model_dir,
        args.labelmap,
        args.score_threshold,
        args.output,
    )
