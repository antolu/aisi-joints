import logging
import os
from argparse import ArgumentParser, Namespace
from os import path
from typing import Dict, Union, Optional
import tensorflow as tf

import pandas as pd
from tqdm import tqdm

from .utils import load_model, load_labelmap, load_image, run_inference, plot_and_save, format_detections

log = logging.getLogger(__name__)


def detect(args: Namespace):
    model = load_model(args.model_dir)
    label_map = load_labelmap(args.labelmap)

    if path.isdir(args.input):
        files = os.listdir(args.input)
        files = [f for f in files if (f.endswith('jpg' or f.endswith('png')))]

        ground_truth = None
    else:
        df = pd.read_csv(args.input)
        files = df['filepath']

        ground_truth = df['cls']

    if not path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    for i, file in tqdm(enumerate(files)):
        image = load_image(file)
        detections = run_inference(image, model)

        boxes = format_detections(detections)

        boxes_filtered = boxes[boxes['detection_scores'] >= args.score_threshold]

        if args.save_plot:
            plot_and_save(image, label_map, detections, args.score_threshold,
                          path.join(args.output, path.split(file)[-1]))


def csv_detect(csv_path: str, model_path: str,
               label_map: Union[Dict[str, int], str], score_threshold: float = 0.5,
               output: Optional[str] = None):
    df = pd.read_csv(csv_path)

    if isinstance(label_map, str) and path.isfile(label_map):
        label_map = load_labelmap(label_map)

    model = load_model(model_path)

    df = dataframe_detect(df, model, label_map, score_threshold)

    if output is not None:
        df.to_csv(output)
    else:
        df.to_csv(csv_path)


def dataframe_detect(df: pd.DataFrame, model: tf.keras.Model,
                     label_map: Dict[int, dict], score_threshold: float = 0.5) \
            -> pd.DataFrame:

    results = []
    for sample in df.itertuples():
        image = load_image(sample.filepath)
        detections = run_inference(image, model)

        boxes = format_detections(detections)

        boxes_filtered = boxes[boxes['detection_scores'] >= score_threshold]

        if len(boxes_filtered) > 1:
            raise RuntimeError('Don\'t know what to do with more than one box.')

        item = boxes_filtered.iloc[0]
        res = {
            'eventId': sample.eventId,
            'detected_class': label_map[item.detection_classes]['name'],
            'detected_x0': item.left,
            'detected_x1': item.right,
            'detected_y0': item.bottom,
            'detected_y1': item.top,
            'detection_score': item.detection_scores
        }

        results.append(res)

    res_df = pd.DataFrame(results)

    df = pd.merge(df, res_df, on='eventId')

    return df


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-i', '--input', required=True,
                        help='Path to .csv containing dataset, '
                             'or path to directory containing images.')
    parser.add_argument('-l', '--labelmap', required=True,
                        help='Path to label map file.')
    parser.add_argument('-m', '--model-dir', dest='model_dir', required=True,
                        help='Path to directory containing exported model.')
    parser.add_argument('-t', '--score-threshold', dest='score_threshold',
                        default=0.5,
                        help='Detection score threshold. All detections under '
                             'this confidence score are discarded.')
    parser.add_argument('-o', '--output', default='output',
                        help='Output directory for images.')
    parser.add_argument('--save-plot', dest='save_plot', action='store_true',
                        help='Save images with bounding boxes.')
    args = parser.parse_args()

    csv_detect(args.input, args.model_dir, args.labelmap, args.score_threshold, args.output)
    # detect(args)

