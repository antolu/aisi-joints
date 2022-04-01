import logging
from typing import Tuple

import pandas as pd
import tensorflow as tf

log = logging.getLogger(__name__)


def get_data(csv_path: str) -> Tuple[tf.data.Dataset,
                                     tf.data.Dataset,
                                     tf.data.Dataset]:
    log.info(f'Reading dataset from {csv_path}.')
    df = pd.read_csv(csv_path)

    # replace string name with integer class
    df.loc[df['cls'] == 'DEFECT', 'cls'] = 0
    df.loc[df['cls'] == 'OK', 'cls'] = 1

    # process splits
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']

    log.info('Read dataset with train/validation/test {}/{}/{} samples.'
             .format(len(train_df), len(val_df), len(test_df)))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath'].to_numpy(), train_df['cls'].to_numpy(dtype=int)))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_df['filepath'].to_numpy(), val_df['cls'].to_numpy(dtype=int)))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df['filepath'].to_numpy(), test_df['cls'].to_numpy(dtype=int)))

    train_dataset = train_dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


@tf.function
def read_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    # tensorflow provides quite a lot of apis for io
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    return image


@tf.function
def normalize(image: tf.Tensor) -> tf.Tensor:
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    image = (2 * image) - 1
    return image


@tf.function
def augment(image: tf.Tensor) -> tf.Tensor:
    # image = tf.image.random_crop(image, (178, 178, 3))
    image = tf.image.resize(image, (299, 299))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, 0.5)
    return image


@tf.function
def preprocess(image_path: str, label: int) -> Tuple[tf.Tensor, int]:
    image = read_image(image_path)
    image = augment(image)
    image = normalize(image)
    return image, label
