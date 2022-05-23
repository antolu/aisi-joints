"""
Write Tensorboard Summary at some validations step
"""
import logging

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

from ._dataloader import process_example
from ..constants import INV_LABEL_MAP

log = logging.getLogger(__name__)


class EvaluateImages:
    def __init__(self, model: tf.keras.Model, dataset_pth: str,
                 tb_writer: tf.summary.SummaryWriter,
                 batch_size: int = 32, log_every: int = 10):
        self._model = model
        self._writer = tb_writer
        self._log_every = log_every

        self._data = tf.data.TFRecordDataset(dataset_pth)
        self._preprocess_data(batch_size)

    def _preprocess_data(self, batch_size: int):
        self._data = self._data.map(
            lambda smpl: process_example(
                smpl, random_crop=False, get_metadata=True,
                augment_data=False))
        self._data = self._data.batch(batch_size)

    def evaluate(self, step: int, num_images: int = 20):
        if step % self._log_every != 0:
            return
        for i, (images, labels, bboxes, eventIds) in enumerate(self._data):
            batch_size = tf.shape(images)[0]
            predictions = self._model(images)

            pred_labels = tf.expand_dims(
                tf.math.argmax(predictions, axis=1), 1)
            gt_labels = tf.expand_dims(tf.math.argmax(labels, axis=1), 1)

            # use makeshift box because the original boxes are destroyed from random cropping
            bboxes = tf.transpose(
                tf.stack(
                    [0.1 * tf.ones(batch_size),
                     0.1 * tf.ones(batch_size),
                     0.9 * tf.ones(batch_size),
                     0.9 * tf.ones(batch_size)]
                )
            )
            bboxes = tf.expand_dims(bboxes, axis=1)
            scores = tf.expand_dims(tf.reduce_max(predictions, axis=1), 1)

            images = tf.cast(images, tf.uint8)
            orig_images = tf.identity(images)

            images = viz_utils.draw_bounding_boxes_on_image_tensors(
                images, bboxes, pred_labels + tf.constant(
                    1, dtype=tf.int64), scores, INV_LABEL_MAP)

            orig_images = viz_utils.draw_bounding_boxes_on_image_tensors(
                orig_images, bboxes, gt_labels + 1, tf.ones_like(scores),
                INV_LABEL_MAP)

            sum_images = tf.concat([images, orig_images], axis=2)
            with self._writer.as_default(step):
                tf.summary.image('Validation image', sum_images,
                                 step=step, max_outputs=num_images)
