"""
Write Tensorboard Summary at some validations step
"""
import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

from .data import process_example
from ..constants import INV_LABEL_MAP

log = logging.getLogger(__name__)


class EvaluateImages:
    def __init__(self, model: tf.keras.Model, dataset_pth: str,
                 batch_size: int = 32):
        self._model = model

        self._data = tf.data.TFRecordDataset(dataset_pth)
        self._preprocess_data(batch_size)

    def _preprocess_data(self, batch_size: int):
        self._data = self._data.map(lambda smpl: process_example(smpl, random_crop=False, get_metadata=True))
        self._data = self._data.batch(batch_size)

    def evaluate(self, step: int, num_images: int = 20,
                 tb_writer: Optional[tf.summary.SummaryWriter] = None):
        for i, (images, labels, bboxes, eventIds) in enumerate(self._data):
            predictions = self._model(images)

            pred_labels = tf.math.argmax(predictions, axis=1)

            bboxes = tf.expand_dims(bboxes, axis=1)
            scores = predictions[pred_labels]

            viz_utils.draw_bounding_boxes_on_image_tensors(
                images, bboxes, pred_labels + tf.constant(1), scores, INV_LABEL_MAP)

            with tb_writer.as_default(step):
                tf.summary.image('Validation image', images, step=step)
