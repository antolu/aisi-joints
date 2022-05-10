from argparse import ArgumentParser, Namespace
import logging

import torch
from datasets import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from transformers import DetrFeatureExtractor

from ..utils.logging import setup_logger
from ..detr._detr import Detr
from ._data import CocoDetection
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


log = logging.getLogger(__name__)


def evaluate(model, dataset):
    base_ds = get_coco_api_from_dataset(
        dataset)  # this is actually just calling the coco attribute
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds,
                                   iou_types)  # initialize evaluator with ground truths

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    feature_extractor = model.feature_extractor

    log.info('Running evaluation...')

    dataloader = DataLoader(base_ds, shuffle=False)

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
        res = {target['image_id'].item(): output for target, output in
               zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


def main(args: Namespace):

    model = Detr.load_from_checkpoint(args.checkpoint_path)

    dataset = CocoDetection(args.data_dir, args.split, model.feature_extractor)

    evaluate(model, dataset)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('data_dir', help='Path to root of COCO format'
                                         'dataset folder.')
    parser.add_argument('-l', '--labelmap', default=None,
                        help='Path to label map file.')
    # parser.add_argument('-m', '--model-dir', dest='model_dir', required=True,
    #                     help='Path to directory containing exported model.')
    parser.add_argument('-c', '--checkpoint-path', dest='checkpoint_path',
                        required=True,
                        help='Path to checkpoint file.')
    parser.add_argument('-s', '--split',
                        choices=['train', 'validation', 'test'],
                        default=None,
                        help='Specific split to evaluate on.')
    parser.add_argument('-t', '--score-threshold', dest='score_threshold',
                        default=0.5,
                        help='Detection score threshold. All detections under '
                             'this confidence score are discarded.')

    args = parser.parse_args()

    setup_logger()
    main(args)
