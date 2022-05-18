import logging
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Dict

import torch
from object_detection.metrics.coco_tools import COCOWrapper
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from ._data import CocoDetection, collate_fn, results_to_coco
from ..detr._detr import Detr
from ..eval.evaluate import evaluate_and_print
from .._utils.logging import setup_logger, time_execution

log = logging.getLogger(__name__)


def filter_results(results: Dict[int, dict], score_threshold: float = 0.7) \
        -> Dict[int, dict]:
    for img_id, res in results.items():
        mask = res['scores'] >= score_threshold
        res['labels'] = res['labels'][mask]
        res['boxes'] = res['boxes'][mask]
        res['scores'] = res['scores'][mask]

    return results


def detect(model: Detr, dataset: CocoDetection, score_threshold: float = 0.7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    feature_extractor = model.feature_extractor

    log.info('Running evaluation...')

    dataloader = DataLoader(dataset, shuffle=False, batch_size=1,
                            collate_fn=partial(collate_fn, feature_extractor))

    all_results = []

    for idx, batch in enumerate(tqdm(dataloader)):
        # get the inputs
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [{k: v.to(device)
                   for k, v in t.items()}
                  for t in batch[
                      'labels']]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack(
            [target['orig_size'] for target in labels], dim=0)
        results = feature_extractor.post_process(outputs,
                                                 orig_target_sizes)  # convert outputs of model to COCO api
        res = {target['image_id'].item():
                   {k: v.detach().to('cpu') for k, v in output.items()}
               for target, output in zip(labels, results)}

        res = filter_results(res, score_threshold)

        all_results.append(res)

    return COCOWrapper(results_to_coco(dataset.coco, all_results))


def main(args: Namespace):
    model = Detr.load_from_checkpoint(args.checkpoint_path,
                                      lr=0.0, lr_backbone=0.0,
                                      weight_decay=0.0, num_classes=2)

    dataset = CocoDetection(args.data_dir, args.split, model.feature_extractor)

    with time_execution() as t:
        detected = detect(model, dataset, args.score_threshold)

    log.info(f'Took {t.duration * 1000 / len(dataset)} ms per sample.')

    evaluate_and_print(dataset.coco, detected)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-d', '--data', dest='data_dir',
                        help='Path to root of COCO format dataset folder.')
    parser.add_argument('-l', '--labelmap', default=None,
                        help='Path to label map file.')
    # parser.add_argument('-m', '--model-dir', dest='model_dir', required=True,
    #                     help='Path to directory containing exported model.')
    parser.add_argument('-c', '--checkpoint-path', dest='checkpoint_path',
                        required=True,
                        help='Path to checkpoint file.')
    parser.add_argument('-s', '--split',
                        choices=['train', 'validation', 'test'],
                        default='test',
                        help='Specific split to evaluate on.')
    parser.add_argument('-t', '--score-threshold', dest='score_threshold',
                        default=0.5, type=float,
                        help='Detection score threshold. All detections under '
                             'this confidence score are discarded.')

    args = parser.parse_args()

    setup_logger()
    main(args)
