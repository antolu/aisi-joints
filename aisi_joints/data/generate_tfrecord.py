"""

Converts .csv with partitioned dataset generated by aisi_joints.data.partition_dataset to .tfrecord
files required by TFOD.

Run as `python -m aisi_joins.data.generate_tfrecord -h` for instructions.
"""

import io
import logging
import os
import os.path as path
from argparse import ArgumentParser, Namespace
from typing import NamedTuple, Dict, Callable, Optional

import pandas as pd
import tensorflow as tf
import tqdm
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from aisi_joints.constants import LABEL_MAP
from .utils import generate_class_weights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)


class Sample(NamedTuple):
    eventId: str
    x0: int
    x1: int
    y0: int
    y1: int
    cls: str
    filepath: str


log = logging.getLogger(__name__)


def read_tfrecord(example: tf.train.Example) -> dict:
    tfrecord_format = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        ),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        ),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        ),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        ),
        'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrecord_format)

    return example


def create_tf_example(
    sample: Sample,
    label_map: Dict[str, int],
    class_weight: Dict[str, float],
    use_class_weights: bool = False,
) -> tf.train.Example:
    # required to find immage dimensions
    with tf.io.gfile.GFile(sample.filepath, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = sample.filepath.encode('utf8')
    image_format = os.path.splitext(sample.filepath)[-1].encode('utf8')

    feature = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(
            [sample.x0 / width]
        ),
        'image/object/bbox/xmax': dataset_util.float_list_feature(
            [sample.x1 / width]
        ),
        'image/object/bbox/ymin': dataset_util.float_list_feature(
            [sample.y0 / height]
        ),
        'image/object/bbox/ymax': dataset_util.float_list_feature(
            [sample.y1 / height]
        ),
        'image/object/class/text': dataset_util.bytes_list_feature(
            [sample.cls.encode('utf8')]
        ),
        'image/object/class/label': dataset_util.int64_list_feature(
            [label_map[sample.cls]]
        ),
    }

    if use_class_weights:
        feature['image/object/weight'] = dataset_util.float_list_feature(
            [class_weight[sample.cls]]
        )

    # convert everything to byte format for tfrecord
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


def generate_tfrecord(
    df: pd.DataFrame,
    label_map: dict,
    output_dir: str,
    use_class_weights: bool = False,
    progress_cb: Optional[Callable] = None,
):

    if 'split' in df:
        splits = df['split'].unique()
        use_splits = True
    else:
        splits = ['samples']
        use_splits = False

    class_weights = generate_class_weights(df['cls'].to_list())
    if use_class_weights:
        log.info(f'Using class weights {class_weights}.')

        msg = [f'Using class weights {class_weights}.']
    else:
        msg = []

    progress = 0
    if progress_cb is None:
        pbar = tqdm.tqdm(total=len(df))

    if output_dir.endswith('.tfrecord'):
        single_output = True
    else:
        single_output = False

    def update_progress():
        if progress_cb is not None:
            nonlocal progress
            progress += 1
            progress_cb(progress, len(df))
        else:
            pbar.update(1)

    def process_df(dataframe: pd.DataFrame, file: str):
        with tf.io.TFRecordWriter(file) as writer:
            for item in dataframe.itertuples():
                tf_sample = create_tf_example(
                    item, label_map, class_weights, use_class_weights
                )
                writer.write(tf_sample.SerializeToString())

                update_progress()

    if not output_dir.endswith('.tfrecord'):
        for split in splits:
            filename = path.join(output_dir, split + '.tfrecord')

            if use_splits:
                # process each data split train/validation/test individually
                split_df = df[df['split'] == split]
            else:
                split_df = df

            process_df(split_df, filename)

            info = (
                f'Successfully created the {split} split TFRecord file: '
                f'at {path.abspath(filename)}'
            )
            log.info(info)
            msg.append(info)
    else:
        process_df(df, output_dir)

        info = f'Successfully TFRecord file: ' f'at {path.abspath(output_dir)}'
        log.info(info)
        msg.append(info)

    return '\n'.join(msg)


def main(args: Namespace):
    with open(args.input) as f:
        df = pd.read_csv(f)

    if args.labelmap:
        label_map = label_map_util.get_label_map_dict(args.labelmap)
    else:
        label_map = LABEL_MAP

    msg = generate_tfrecord(df, label_map, args.output, args.balance)
    log.info(msg)


if __name__ == '__main__':
    from .._utils.logging import setup_logger

    parser = ArgumentParser()
    parser.add_argument(
        '-l', '--labelmap', type=str, help='Labelmap in pbtxt format.'
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Input .csv files generated by the preprocess_csv ' 'script.',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='.',
        help='Directory to output TFRecord (.tfrecord) files. '
        'Set to *.tfrecord to only output one record.',
    )
    parser.add_argument(
        '-b',
        '--balance',
        action='store_true',
        help='Balance the dataset by setting class weights',
    )

    args = parser.parse_args()

    setup_logger()
    main(args)
