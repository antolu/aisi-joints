""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import io
import logging
import os
from typing import NamedTuple

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
from PIL import Image
import os.path as path
from argparse import ArgumentParser, Namespace
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import tqdm


class Sample(NamedTuple):
    eventId: str
    x0: int
    x1: int
    y0: int
    y1: int
    cls: str
    filepath: str


log = logging.getLogger(__name__)


def create_tf_example(sample: Sample, label_map: dict) -> tf.train.Example:
    with tf.io.gfile.GFile(sample.filepath, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = sample.filepath.encode('utf8')
    image_format = os.path.splitext(sample.filepath)[-1].encode('utf8')

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature([sample.x0]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([sample.x1]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([sample.y0]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([sample.y1]),
        'image/object/class/text': dataset_util.bytes_list_feature(
            [sample.cls.encode('utf8')]),
        'image/object/class/label': dataset_util.int64_list_feature(
            [label_map[sample.cls]]),
    }))
    return tf_example


def main(args: Namespace):
    label_map = label_map_util.get_label_map_dict(args.labelmap)

    with open(args.input) as f:
        df = pd.read_csv(f)

    splits = df['split'].unique()
    pbar = tqdm.tqdm(total=len(df))
    for split in splits:
        filename = path.join(args.output, split + '.tfrecord')

        split_df = df[df['split'] == split]

        with tf.io.TFRecordWriter(filename) as writer:
            for item in split_df.itertuples():
                tf_sample = create_tf_example(item, label_map)
                writer.write(tf_sample.SerializeToString())

                pbar.update(1)

        log.info(f'Successfully created the {split} split TFRecord file: '
                 f'at {path.abspath(filename)}')


if __name__ == '__main__':
    from ..utils.logging import setup_logger
    parser = ArgumentParser()
    parser.add_argument('-l', '--labelmap', type=str,
                        help='Labelmap in pbtxt format.')
    parser.add_argument('-i', '--input', type=str,
                        help='Input .csv files generated by the preprocess_csv '
                             'script.')
    parser.add_argument('-o', '--output', type=str, default='.',
                        help='Directory to output TFRecord (.record) files.')

    args = parser.parse_args()

    setup_logger()
    main(args)
