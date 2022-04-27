from typing import List

import pandas as pd
import logging
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class JointDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self._data = df

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self):
        pass

    @staticmethod
    def from_csv(csv_path: str) -> 'JointDataset':
        df = pd.read_csv(csv_path)

        return JointDataset(df)


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
    x_offset_low = -min(x0, 0)
    y_offset_low = -min(y0, 0)

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
    x_offset_up = max(max_x, x1) - max_x
    y_offset_up = max(max_y, y1) - max_y

    x0 -= x_offset_up
    x1 -= x_offset_up
    y0 -= y_offset_up
    y1 -= y_offset_up

    return [x0, x1, y0, y1]


def random_crop_bbox(image: torch.Tensor, bndbox: List[int],
                     width: int = 299, height: int = 299) -> torch.Tensor:
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
    max_x, max_y = image.shape[1], image.shape[0]

    crop_width = bndbox[1] - bndbox[0]
    crop_height = bndbox[3] - bndbox[2]

    offset_x = torch.rand(1, 0, width - crop_width)
    offset_y = torch.rand(1, 0, height - crop_height)

    log.debug(f'Sampled offsets: x: {offset_x}, y: {offset_y}')

    log.debug(f'Original bounding box: {bndbox}')

    x0, x1 = bndbox[0] - offset_x, bndbox[1] + (width - crop_width - offset_x)
    y0, y1 = bndbox[2] - offset_y, bndbox[3] + (height - crop_height - offset_y)

    box = [x0, x1, y0, y1]

    # box = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), box))
    box = shift_upper(box, max_x, max_y)
    # box = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), box))
    box = shift_lower(box)
    log.debug(f'Updated bounding box: {box}')

    return crop_and_pad(image, box, width, height)


def center_crop_bbox(image: torch.Tensor, bndbox: list, width: int = 299, height: int = 299) -> torch.Tensor:
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
    y_max, x_max = image.shape[0], image.shape[1]

    crop_width = bndbox[1] - bndbox[0]
    crop_height = bndbox[3] - bndbox[2]

    x0, x1, y0, y1 = bndbox
    x0 -= torch.floor((width - crop_width) / 2)
    x1 += torch.ceil((width - crop_width) / 2)
    y0 -= torch.floor((height - crop_height) / 2)
    y1 += torch.ceil((height - crop_height) / 2)

    # x0 = tf.cast(x0, tf.int32)
    # x1 = tf.cast(x1, tf.int32)
    # y0 = tf.cast(y0, tf.int32)
    # y1 = tf.cast(y1, tf.int32)
    box = shift_upper([x0, x1, y0, y1], x_max, y_max)
    box = shift_lower(box)

    return crop_and_pad(image, box, width, height)


def crop_and_pad(image: torch.Tensor, bndbox: List[int],
                 width: int = 299, height: int = 299) -> torch.Tensor:
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

    # bndbox = list(map(lambda x: tf.squeeze(tf.cast(x, tf.int32)), bndbox))
    x0, x1, y0, y1 = bndbox

    image = image[y0:y1, x0:x1, :]
    image = tf.image.pad_to_bounding_box(image, 0, 0, height, width)

    return image
