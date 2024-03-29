"""
This module provides everything needed for data loading and pre-processing
for the image classification approach.
"""
import logging
import math
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv

from ..constants import LABEL_MAP
from ..data.common import Sample
from ..data.generate_tfrecord import read_tfrecord

log = logging.getLogger(__name__)


class JointsSequence(tf.keras.utils.Sequence):
    """
    Implements a Keras Sequence dataset, which will allow the fit API to load
    multiple batches in parallel.
    """

    def __init__(
        self,
        csv_path_or_df: Union[str, pd.DataFrame],
        split: Optional[str] = None,
        crop_width: int = 299,
        crop_height: int = 299,
        batch_size: int = 32,
        random_crop: bool = True,
        augment_data: bool = True,
        adaptive_threshold: bool = False,
    ):

        if isinstance(csv_path_or_df, str):
            with open(csv_path_or_df, 'r') as f:
                df = pd.read_csv(f)

            if split is not None:
                df = df[df['split'] == split]
        elif isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            raise ValueError

        self._df = df

        self._crop_width = crop_width
        self._crop_height = crop_height
        self._batch_size = batch_size
        self._random_crop = random_crop
        self._augment_data = augment_data
        self._adaptive_threshold = adaptive_threshold

    def __len__(self) -> int:
        return math.ceil(len(self._df) / self._batch_size)

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:

        images = []
        labels = []

        low = index * self._batch_size
        high = (index + 1) * self._batch_size
        for i in range(low, high if high < len(self._df) else len(self._df)):
            sample = Sample.from_dataframe(self._df.iloc[i])
            image, label = self._load_sample(sample)

            images.append(image)
            labels.append(label)

        return tf.stack(images, axis=0), tf.stack(labels)

    def on_epoch_end(self):
        pass

    def _load_sample(self, sample: Sample) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Loads a single sample from disk.

        Parameters
        ----------
        sample: Sample
            Data required to load the sample.

        Returns
        -------
        tf.Tensor, tf.Tensor
            image and label
        """
        image = read_image(sample.filepath, 'png', self._adaptive_threshold)

        image = preprocess(
            image,
            sample.bbox.to_pascal_voc(),
            self._crop_width,
            self._crop_height,
            self._random_crop,
            self._augment_data,
        )

        label = tf.one_hot(LABEL_MAP[sample.bbox.cls] - 1, 2)

        return image, label


def shift_lower(bndbox: List[int]) -> List[int]:
    """
    Shift bounding box upwards, if lower bounds are negative
    (out of bounds)

    Parameters
    ----------
    bndbox : List[int]
        Bounding box in the shape [x0, x1, y0, y1]

    Returns
    -------
    Updated bounding box in the same shape as input.
    """
    x0, x1, y0, y1 = bndbox
    # shift box in case of negative values
    x_offset_low = -tf.math.minimum(x0, tf.constant(0))
    y_offset_low = -tf.math.minimum(y0, tf.constant(0))

    x0 += x_offset_low
    x1 += x_offset_low
    y0 += y_offset_low
    y1 += y_offset_low

    return [x0, x1, y0, y1]


def shift_upper(bndbox: List[int], max_x: int, max_y: int) -> List[int]:
    """
    Shift bounding box downwards, if upper bounds are beyond image edge
    (out of bounds)

    Parameters
    ----------
    bndbox : List[int]
        Bounding box in the shape [x0, x1, y0, y1].
    max_x : int
        Image width
    max_y : int
        Image height

    Returns
    -------
    Updated bounding box in the same shape as input.
    """

    x0, x1, y0, y1 = bndbox
    # shift box in case of OOB values
    log.debug(f'max x: {max_x}, x1: {x1}')
    x_offset_up = tf.math.maximum(max_x, x1) - max_x
    y_offset_up = tf.math.maximum(max_y, y1) - max_y

    x0 -= x_offset_up
    x1 -= x_offset_up
    y0 -= y_offset_up
    y1 -= y_offset_up

    return [x0, x1, y0, y1]


def random_crop_bbox(
    image: tf.Tensor,
    bndbox: List[tf.Tensor],
    width: int = 299,
    height: int = 299,
) -> tf.Tensor:
    """
    Random crop an area around a bounding box to a fixed size.
    If output size is greater than maximum crop size the image will
    be zero-padded.

    Parameters
    ----------
    image : np.ndarray
        image array of size [height, width, 3]
    bndbox : List[int]
        Bounding box in the shape [x0, x1, y0, y1].
    width : int
        Cropped image width
    height : int
        Cropped image height

    Returns
    -------
    Crop of image.
    """
    max_x, max_y = tf.shape(image)[1], tf.shape(image)[0]

    crop_width = bndbox[1] - bndbox[0]
    crop_height = bndbox[3] - bndbox[2]

    offset_x = tf.random.uniform(
        [1], 0, tf.reshape(width - crop_width, []), dtype=tf.int64
    )
    offset_y = tf.random.uniform(
        [1], 0, tf.reshape(height - crop_height, []), dtype=tf.int64
    )

    log.debug(f'Sampled offsets: x: {offset_x}, y: {offset_y}')

    log.debug(f'Original bounding box: {bndbox}')

    x0, x1 = bndbox[0] - offset_x, bndbox[1] + (width - crop_width - offset_x)
    y0, y1 = bndbox[2] - offset_y, bndbox[3] + (
        height - crop_height - offset_y
    )

    box = [x0, x1, y0, y1]

    box = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), box))
    box = shift_upper(box, max_x, max_y)
    box = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), box))
    box = shift_lower(box)
    log.debug(f'Updated bounding box: {box}')

    return crop_and_pad(image, box, width, height)


def center_crop_bbox(
    image: tf.Tensor, bndbox: list, width: int = 299, height: int = 299
) -> tf.Tensor:
    """
    Center crop an area around a bounding box to a fixed size.
    If output size is greater than maximum crop size the image will
    be zero-padded.

    Parameters
    ----------
    image : tf.Tensor
        image array of size [height, width, 3]
    bndbox : List[int]
        Bounding box in the shape [x0, x1, y0, y1].
    width : int
        Cropped image width
    height : int
        Cropped image height

    Returns
    -------
    Crop of image.
    """
    y_max, x_max = tf.shape(image)[0], tf.shape(image)[1]

    crop_width = bndbox[1] - bndbox[0]
    crop_height = bndbox[3] - bndbox[2]

    x0, x1, y0, y1 = bndbox
    x0 -= tf.cast(tf.math.floor((width - crop_width) / 2), tf.int64)
    x1 += tf.cast(tf.math.ceil((width - crop_width) / 2), tf.int64)
    y0 -= tf.cast(tf.math.floor((height - crop_height) / 2), tf.int64)
    y1 += tf.cast(tf.math.ceil((height - crop_height) / 2), tf.int64)

    x0 = tf.cast(x0, tf.int32)
    x1 = tf.cast(x1, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    y1 = tf.cast(y1, tf.int32)
    box = shift_upper([x0, x1, y0, y1], x_max, y_max)
    box = shift_lower(box)

    return crop_and_pad(image, box, width, height)


def crop_and_pad(
    image: tf.Tensor, bndbox: List[int], width: int = 299, height: int = 299
) -> tf.Tensor:
    """
    Crop image to specific size using bounding box,
    zero-pad if crop is too large.

    Parameters
    ----------
    image : tf.Tensor
        image array of size [height, width, 3]
    bndbox : List[int]
        Bounding box in the shape [x0, x1, y0, y1].
    width : int
        Cropped image width
    height : int
        Cropped image height

    Returns
    -------
    Crop of image.
    """

    bndbox = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), bndbox))
    x0, x1, y0, y1 = bndbox

    image = image[y0:y1, x0:x1, :]
    image = tf.image.pad_to_bounding_box(image, 0, 0, height, width)

    return image


def read_image(
    image_path: str,
    fmt: Optional[str] = None,
    adaptive_threshold: bool = False,
) -> tf.Tensor:
    image = tf.io.read_file(image_path)

    if fmt == 'png':
        image = tf.image.decode_png(image, channels=3)
    elif fmt == 'jpeg':
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_image(image, channels=3)

    # does only work in eager mode
    if adaptive_threshold:
        image = image.numpy().astype(np.uint8)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.adaptiveThreshold(
            image,
            255.0,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            15,
            3,
        )
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        image = tf.convert_to_tensor(image, tf.float32)
    else:
        image = tf.cast(image, tf.float32)

    return image


def augment(image: tf.Tensor) -> tf.Tensor:
    """
    Augments 3-channel image tensors.

    Parameters
    ----------
    image: tf.Tensor

    Returns
    -------
    tf.Tensor
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.0, 0.8)
    image = tf.image.random_saturation(image, 0.0, 4.0)
    image = tf.image.random_brightness(image, 0.5)
    return image


def preprocess(
    image: tf.Tensor,
    bbox: List[tf.Tensor],
    width: int = 299,
    height: int = 299,
    random_crop: bool = True,
    augment_data: bool = True,
    preprocess_fn: Callable = None,
):
    if random_crop:
        image = random_crop_bbox(image, bbox, width, height)
    else:
        image = center_crop_bbox(image, bbox, width, height)

    if augment_data:
        image = augment(image)
    if preprocess_fn is not None:
        image = preprocess_fn(image)

    return image
