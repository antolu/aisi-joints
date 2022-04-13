import logging
import math
from functools import partial
from typing import Tuple, List

import numpy as np
import tensorflow as tf

from aisi_joints.data.generate_tfrecord import read_tfrecord

log = logging.getLogger(__name__)


def load_tfrecord(pth: str, batch_size: int,
                  random_crop: bool = True,
                  shuffle: bool = True,
                  augment_data: bool = True) -> tf.data.TFRecordDataset:
    data = tf.data.TFRecordDataset(pth)

    if shuffle:
        data = data.shuffle(2048, reshuffle_each_iteration=True)
    data = data.map(lambda smpl: process_example(smpl, random_crop, augment_data=augment_data))
    data = data.batch(batch_size)

    return data


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


def random_crop_bbox(image: tf.Tensor, bndbox: List[int],
                     width: int = 299, height: int = 299) -> tf.Tensor:
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

    # if crop_width > width:
    #     print(crop_width.numpy())
    #     raise ValueError(f'Crop is wider than max image width: {crop_width} > {width}.')
    # if crop_height > height:
    #     raise ValueError(f'Crop is higher than max image height: {crop_height} > {height}.')

    offset_x = tf.random.uniform([1], 0, tf.reshape(width - crop_width, []), dtype=tf.int64)
    offset_y = tf.random.uniform([1], 0, tf.reshape(height - crop_height, []), dtype=tf.int64)

    log.debug(f'Sampled offsets: x: {offset_x}, y: {offset_y}')

    log.debug(f'Original bounding box: {bndbox}')

    x0, x1 = bndbox[0] - offset_x, bndbox[1] + (width - crop_width - offset_x)
    y0, y1 = bndbox[2] - offset_y, bndbox[3] + (height - crop_height - offset_y)

    box = [x0, x1, y0, y1]

    box = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), box))
    box = shift_upper(box, max_x, max_y)
    box = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), box))
    box = shift_lower(box)
    log.debug(f'Updated bounding box: {box}')

    return crop_and_pad(image, box, width, height)


def center_crop_bbox(image: np.ndarray, bndbox: list, width: int = 299, height: int = 299) -> np.ndarray:
    """
    Center crop an area around a bounding box to a fixed size.
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
    y_max, x_max = tf.shape(image)[1], tf.shape(image)[0]

    crop_width = bndbox[1] - bndbox[0]
    crop_height = bndbox[3] - bndbox[2]

    # if crop_width > width:
    #     raise ValueError('Crop is wider than max image width')
    # if crop_height > height:
    #     raise ValueError('Crop is higher than max image height')

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


def crop_and_pad(image: tf.Tensor, bndbox: List[int],
                 width: int = 299, height: int = 299) -> tf.Tensor:
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
    y_max, x_max = tf.shape(image)[0], tf.shape(image)[1]
    # for i in tf.range(len(bndbox)):
    #     bndbox[i] = int(bndbox[i])

    bndbox = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), bndbox))
    x0, x1, y0, y1 = bndbox

    image = image[y0:y1, x0:x1, :]

    x_pad = -tf.math.minimum(tf.constant(0), x_max - x1)
    y_pad = -tf.math.minimum(tf.constant(0), y_max - y1)

    # zero-pad if crop is too large
    image = tf.pad(image, [[0, y_pad], [0, x_pad], [0, 0]])

    return image


# @tf.function
def read_image(image_path: str, fmt: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    # tensorflow provides quite a lot of apis for io

    if fmt == 'png':
        image = tf.image.decode_png(image, channels=3)
    elif fmt == 'jpeg':
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


# @tf.function
def normalize(image: tf.Tensor) -> tf.Tensor:
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    image = (2 * image) - 1
    return image


# @tf.function
def augment(image: tf.Tensor, bbox: List[int], random_crop: bool = True) -> tf.Tensor:
    # image = tf.image.random_crop(image, (178, 178, 3))
    # if random_crop:
    #     image = random_crop_bbox(image, bbox, 299, 299)
    # else:
    #     image = center_crop_bbox(image, bbox, 299, 299)
    # image = tf.image.resize(image, (299, 299))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, 0.5)
    return image


# @tf.function
def preprocess(image_path: str, label: int, fmt: str, bbox: List[int],
               random_crop: bool = True) -> Tuple[tf.Tensor, int]:
    image = read_image(image_path, fmt)
    image = augment(image, bbox, random_crop=random_crop)
    # image = normalize(image)
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    return image, tf.one_hot(label - 1, 2)


@tf.function
def process_example(data: tf.train.Example, random_crop: bool = True,
                    get_metadata: bool = False, augment_data: bool = True):
    sample = read_tfrecord(data)

    image_path = sample['image/filename']
    label = sample['image/object/class/label']

    height = tf.cast(sample['image/height'], tf.float32)
    width = tf.cast(sample['image/width'], tf.float32)
    bbox = [
        sample['image/object/bbox/xmin'] * width,
        sample['image/object/bbox/xmax'] * width,
        sample['image/object/bbox/ymin'] * height,
        sample['image/object/bbox/ymax'] * height,
    ]
    bbox = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int64)), bbox))

    fmt = sample['image/format']

    image = read_image(image_path, fmt)

    if random_crop:
        image = random_crop_bbox(image, bbox, 299, 299)
    else:
        image = center_crop_bbox(image, bbox, 299, 299)

    if augment_data:
        image = augment(image, bbox, random_crop=random_crop)
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

    labels = tf.one_hot(label - 1, 2)

    if not get_metadata:
        return image, labels
    else:
        return image, labels, bbox, sample['image/source_id']


def undo_preprocess(image: tf.Tensor) -> tf.Tensor:
    image += 1
    image *= 127.5

    return image