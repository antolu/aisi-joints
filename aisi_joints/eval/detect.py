import logging
import os
from argparse import ArgumentParser, Namespace
from os import path

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

    detect(args)
