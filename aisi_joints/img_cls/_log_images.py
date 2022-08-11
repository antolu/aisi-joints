"""
This module contains the EvaluateImages class that can be used as a callback
during validation to perform inference using a dataset and display predicted
vs ground truth labeling side by side in TensorBoard.
"""
import logging
from typing import Union

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

from ._dataloader import JointsSequence
from ..constants import INV_LABEL_MAP

log = logging.getLogger(__name__)


class EvaluateImages:
    def __init__(
        self,
        model: tf.keras.Model,
        val_data: JointsSequence,
        tb_writer: tf.summary.SummaryWriter,
        log_every: int = 10,
    ):
        self._model = model
        self._writer = tb_writer
        self._log_every = log_every

        self._data = val_data

    def evaluate(self, step: int, num_images: int = 20):
        if step % self._log_every != 0:
            return
        for i, (images, labels) in enumerate(self._data):
            batch_size = tf.shape(images)[0]
            predictions = self._model(images, training=False)

            pred_labels = tf.expand_dims(
                tf.math.argmax(predictions, axis=1), 1
            )
            gt_labels = tf.expand_dims(tf.math.argmax(labels, axis=1), 1)

            # use makeshift box because the original boxes are destroyed from random cropping
            bboxes = tf.transpose(
                tf.stack(
                    [
                        0.1 * tf.ones(batch_size),
                        0.1 * tf.ones(batch_size),
                        0.9 * tf.ones(batch_size),
                        0.9 * tf.ones(batch_size),
                    ]
                )
            )
            bboxes = tf.expand_dims(bboxes, axis=1)
            scores = tf.expand_dims(tf.reduce_max(predictions, axis=1), 1)

            images = tf.cast(images, tf.uint8)
            orig_images = tf.identity(images)

            images = viz_utils.draw_bounding_boxes_on_image_tensors(
                images,
                bboxes,
                pred_labels + tf.constant(1, dtype=tf.int64),
                scores,
                INV_LABEL_MAP,
            )

            orig_images = viz_utils.draw_bounding_boxes_on_image_tensors(
                orig_images,
                bboxes,
                gt_labels + 1,
                tf.ones_like(scores),
                INV_LABEL_MAP,
            )

            sum_images = tf.concat([images, orig_images], axis=2)
            with self._writer.as_default(step):
                tf.summary.image(
                    'Validation image',
                    sum_images,
                    step=step,
                    max_outputs=num_images,
                )
