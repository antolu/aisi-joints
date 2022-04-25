from argparse import ArgumentParser, Namespace
from typing import List

from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os
import cv2 as cv
import random
from os import path
import torch
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# !python d2/converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth

from d2.train_net import Trainer
from d2.detr import add_detr_config


def filter_predictions_from_outputs(outputs: torch.Tensor,
                                    threshold: float = 0.7,
                                    verbose: bool = True):
    predictions = outputs["instances"].to("cpu")

    if verbose:
        print(list(predictions.get_fields()))

    indices = [i
               for (i, s) in enumerate(predictions.scores)
               if s >= threshold
               ]

    filtered_predictions = predictions[indices]

    return filtered_predictions


def register_datasets(dataset_root: str) -> List[str]:
    ann_dir = path.join(dataset_root, 'annotations')
    if not path.exists(ann_dir):
        raise NotADirectoryError(f'Could not find directory {ann_dir}.')

    splits = []

    for split_json in os.listdir(ann_dir):
        split_name = path.splitext(split_json)[0]

        split_imgs = path.join(dataset_root, split_name)
        if not path.isdir(split_imgs):
            raise NotADirectoryError(f'Could not find directory {split_imgs} '
                                     f'for split {split_name}.')

        register_coco_instances(split_name, {}, path.join(ann_dir, split_json),
                                split_imgs)

        splits.append(split_name)

    return splits


def main(args: Namespace):
    cfg = get_cfg()

    add_detr_config(cfg)
    cfg.merge_from_file(args.config)

    register_datasets(args.dataset)

    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)
    #
    # threshold = 0.7
    #
    # filtered_predictions = filter_predictions_from_outputs(outputs,
    #                                                        threshold=threshold)
    #
    # # We can use `Visualizer` to draw the predictions on the image.
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(filtered_predictions)
    # cv2.imshow(out.get_image()[:, :, ::-1])

    cat_names = ['OK', 'DEFECT']

    for keyword in ['train', 'validation']:
        MetadataCatalog.get(keyword).set(thing_classes=cat_names)

    custom_metadata = MetadataCatalog.get('train')

    dataset_dicts = DatasetCatalog.get('train')
    for d in random.sample(dataset_dicts, 3):
        img = cv.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata,
                                scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        # cv.imshow(d['file_name'], out.get_image()[:, :, ::-1])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("validation", None, False,
                              output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, 'validation')
    results = inference_on_dataset(trainer.model, val_loader, evaluator)

    print(results)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('config', type=str,
                        help='Path to detectron config file.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset directory.')

    args = parser.parse_args()

    main(args)

