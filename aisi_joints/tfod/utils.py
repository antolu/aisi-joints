import logging
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

log = logging.getLogger(__name__)


def load_model(pth: str) -> tf.keras.Model:
    log.debug("Loading model...")

    start_time = time.time()

    # Load saved model and build the detection function
    model = tf.saved_model.load(pth)

    end_time = time.time()
    elapsed_time = end_time - start_time
    log.debug("Done! Took {} seconds".format(elapsed_time))

    return model


def load_labelmap(pth: str) -> Dict[str, int]:
    category_index = label_map_util.create_category_index_from_labelmap(
        pth, use_display_name=True
    )

    return category_index


def load_image(pth: str) -> np.ndarray:
    return np.array(Image.open(pth))


def remove_duplicate_boxes(detections):
    nms_indices = tf.image.non_max_suppression(
        detections["detection_boxes"], detections["detection_scores"], 40
    )

    detections["detection_scores"] = tf.gather(
        detections["detection_scores"], nms_indices, axis=0
    )
    detections["detection_boxes"] = tf.gather(
        detections["detection_boxes"], nms_indices, axis=0
    )
    detections["detection_classes"] = tf.gather(
        detections["detection_classes"], nms_indices, axis=0
    )

    return detections


def scale_boxes_to_img_size(img: np.ndarray, boxes_info: dict) -> np.ndarray:
    height, width, _ = img.shape
    boxes_info["detection_boxes"] = boxes_info["detection_boxes"].numpy()

    scaled_boxes = [
        [height * bbx[0], width * bbx[1], height * bbx[2], width * bbx[3]]
        for bbx in boxes_info["detection_boxes"]
    ]

    scaled_boxes = np.reshape(scaled_boxes, (-1, 4))

    return scaled_boxes


def run_inference(image: np.ndarray, model: tf.keras.Model) -> dict:
    tensor = tf.convert_to_tensor(image)

    tensor = tensor[tf.newaxis, ...]

    detections = model(tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    filtered_detections = remove_duplicate_boxes(detections)
    filtered_detections["detection_boxes"] = scale_boxes_to_img_size(
        image, filtered_detections
    )

    return detections


def format_detections(detections: dict) -> pd.DataFrame:
    detection_boxes_int = detections["detection_boxes"]

    boxes_info_dict = {
        "detection_classes": detections["detection_classes"],
        "detection_scores": detections["detection_scores"],
        "left": np.rint(detection_boxes_int[:, 1]),
        "bottom": np.rint(detection_boxes_int[:, 0]),
        "right": np.rint(detection_boxes_int[:, 3]),
        "top": np.rint(detection_boxes_int[:, 2]),
    }

    boxes_info_df = pd.DataFrame.from_records(
        boxes_info_dict,
        columns=[
            "detection_classes",
            "detection_scores",
            "left",
            "bottom",
            "right",
            "top",
        ],
    )

    return boxes_info_df


def plot_and_save(
    image: np.ndarray,
    label_map: dict,
    detections: dict,
    score_threshold: float,
    output_path: str,
):
    image_copy = np.array(image)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_copy,
        np.array(detections["detection_boxes"]),
        np.array(detections["detection_classes"]),
        np.array(detections["detection_scores"]),
        label_map,
        use_normalized_coordinates=False,
        max_boxes_to_draw=40,
        min_score_thresh=score_threshold,
        agnostic_mode=False,
    )

    plt.imsave(output_path, image_copy)
