"""
Convenience scrip to launch fiftyone application for a processed csv file.
Must be run as a python module to avoid naming conflict
    python -m aisi_joints.fiftyone
"""
import logging
from argparse import Namespace, ArgumentParser

import fiftyone as fo
import pandas as pd
from PIL import Image
from fiftyone.utils.voc import VOCBoundingBox

log = logging.getLogger(__name__)


def create_sample(df: pd.DataFrame) -> fo.Sample:
    metadata_fields = ('eventId', 'sessionId', 'fingerprintId', 'platformId')

    sample = fo.Sample(filepath=df.filepath)
    image = Image.open(df.filepath)

    bbox = VOCBoundingBox(df.x0, df.y0, df.x1, df.y1)
    ground_truth = fo.Detections(
        detections=[
            fo.Detection(label=df.cls, bounding_box=bbox.to_detection_format(image.size))]
    )
    sample['ground_truth'] = ground_truth

    if hasattr(df, 'detected_class'):
        detected_bbox = VOCBoundingBox(df.detected_x0, df.detected_y0,
                                       df.detected_x1, df.detected_y1)
        prediction = fo.Detections(
            detections=[
                fo.Detection(label=df.detected_class,
                             bounding_box=detected_bbox.to_detection_format(image.size),
                             confidence=df.detection_score)
            ]
        )
        sample['prediction'] = prediction

    for metadata in metadata_fields:
        if hasattr(df, metadata):
            sample[metadata] = getattr(df, metadata)

    return sample


def main(args: Namespace):
    df = pd.read_csv(args.csv_path)

    dataset: fo.Dataset = fo.Dataset()

    samples = []
    for row in df.itertuples():
        sample = create_sample(row)
        samples.append(sample)

        dataset.add_sample(sample)

    session = fo.launch_app()
    session.dataset = dataset
    session.wait()
