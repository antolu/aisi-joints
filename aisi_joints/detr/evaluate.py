"""
This module provides a script to evaluate a trained DE:TR model.

The evaluation uses TFOD functions to calculate evaluation metrics.

This module is runnable. Use the `-h` option to view usage.
"""
import logging
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Dict, List, Optional

import torch
from object_detection.metrics.coco_tools import COCOWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._data import CocoDetection
from ._data import collate_fn
from ._data import results_to_coco
from .._utils.logging import setup_logger
from .._utils.utils import time_execution
from ..detr._detr import Detr
from ..tfod.evaluate import evaluate_and_print

log = logging.getLogger(__name__)


def filter_results(
    results: Dict[int, dict], score_threshold: float = 0.7
) -> Dict[int, dict]:
    """
    Filter out results from a batch prediction output that is below the score
    threshold.

    Parameters
    ----------
    results: Dict[int, dict]
        Mapping from img id (COCO) to dictionary with results.
    score_threshold: float
        Score threshold for filtering results.

    Returns
    -------
    Dict[int, dict]
        Same dictionary as input, with results under score_threshold removed.
    """
    for img_id, res in results.items():
        mask = res['scores'] >= score_threshold
        res['labels'] = res['labels'][mask]
        res['boxes'] = res['boxes'][mask]
        res['scores'] = res['scores'][mask]

    return results


def detect(
    model: Detr, dataset: CocoDetection, score_threshold: float = 0.7
) -> COCOWrapper:
    """
    Runs detection using a DE:TR model and filters detections using a
    threshold.

    Parameters
    ----------
    model: Detr
        Trained, or loaded from checkpoint, DE:TR model.
    dataset: CocoDetection
        A dataset to run detections on. N.B. this should be a subclass of
        torchvision CocoDetection
    score_threshold: float
        Threshold to filter detections with.

    Returns
    -------
    COCOWrapper
        A COCOWrapper ready to be evaluated using pycocotools metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    feature_extractor = model.feature_extractor

    log.info('Running evaluation...')

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=partial(collate_fn, feature_extractor),
    )

    all_results = []

    for idx, batch in enumerate(tqdm(dataloader)):
        # get the inputs
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [
            {k: v.to(device) for k, v in t.items()} for t in batch['labels']
        ]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack(
            [target['orig_size'] for target in labels], dim=0
        )
        results = feature_extractor.post_process(
            outputs, orig_target_sizes
        )  # convert outputs of model to COCO api
        res = {
            target['image_id'].item(): {
                k: v.detach().to('cpu') for k, v in output.items()
            }
            for target, output in zip(labels, results)
        }

        res = filter_results(res, score_threshold)

        all_results.append(res)

    return COCOWrapper(results_to_coco(dataset.coco, all_results))


def evaluate(args: Namespace):
    model = Detr.load_from_checkpoint(
        args.checkpoint_path,
        lr=0.0,
        lr_backbone=0.0,
        weight_decay=0.0,
        momentum=0.0,
        num_classes=2,
    )

    dataset = CocoDetection(args.data_dir, args.split, model.feature_extractor)

    with time_execution() as t:
        detected = detect(model, dataset, args.score_threshold)

    log.info(f'Took {t.duration * 1000 / len(dataset)} ms per sample.')

    evaluate_and_print(dataset.coco, detected)


def main(argv: Optional[List[str]] = None):
    parser = ArgumentParser()

    parser.add_argument(
        '-d',
        '--data',
        dest='data_dir',
        help='Path to root of COCO format dataset folder.',
    )
    parser.add_argument(
        '-l', '--labelmap', default=None, help='Path to label map file.'
    )
    # parser.add_argument('-m', '--model-dir', dest='model_dir', required=True,
    #                     help='Path to directory containing exported model.')
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        dest='checkpoint_path',
        required=True,
        help='Path to checkpoint file.',
    )
    parser.add_argument(
        '-s',
        '--split',
        choices=['train', 'validation', 'test'],
        default='test',
        help='Specific split to evaluate on.',
    )
    parser.add_argument(
        '-t',
        '--score-threshold',
        dest='score_threshold',
        default=0.7,
        type=float,
        help='Detection score threshold. All detections under '
        'this confidence score are discarded.',
    )

    args = parser.parse_args(argv)

    setup_logger()
    evaluate(args)


if __name__ == '__main__':
    main()
