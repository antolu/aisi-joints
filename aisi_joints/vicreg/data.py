from typing import List, Tuple, Callable, Any, Optional

import pandas as pd
import logging
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from ..data.common import Sample
from ..constants import LABEL_MAP

log = logging.getLogger(__name__)


class JointDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df: pd.DataFrame, random_crop: bool = False,
                 crop_width: int = 256, crop_height: int = 256,
                 transform: Callable[[Any], torch.Tensor] = None):
        super().__init__()

        self._data = df
        self._random_crop = random_crop
        self._crop_width = crop_width
        self._crop_height = crop_height

        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx > len(self) - 1:
            raise IndexError
        sample = Sample.from_dataframe(self._data.iloc[idx])
        image = np.asarray(Image.open(sample.filepath))

        bbox = sample.bbox.to_pascal_voc()

        if self._random_crop:
            image = random_crop_bbox(image, bbox, self._crop_width,
                                     self._crop_height)
        else:
            image = center_crop_bbox(image, bbox, self._crop_width,
                                     self._crop_height)

        label = LABEL_MAP[sample.bbox.cls] - 1

        image = Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)

        return image, label

    @staticmethod
    def from_csv(csv_path: str, random_crop: bool = False,
                 crop_width: int = 256, crop_height: int = 256,
                 transform: Optional[Callable[[Any], torch.Tensor]] = None) \
            -> 'JointDataset':
        df = pd.read_csv(csv_path)

        return JointDataset(df, random_crop, crop_width, crop_height,
                            transform)


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
    x_offset_up = max(max_x, x1) - max_x
    y_offset_up = max(max_y, y1) - max_y

    x0 -= x_offset_up
    x1 -= x_offset_up
    y0 -= y_offset_up
    y1 -= y_offset_up

    return [x0, x1, y0, y1]


def random_crop_bbox(image: np.ndarray, bndbox: List[int],
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

    offset_x = torch.randint(0, width - crop_width, (1,)).item()
    offset_y = torch.randint(0, height - crop_height, (1,)).item()

    log.debug(f'Sampled offsets: x: {offset_x}, y: {offset_y}')

    log.debug(f'Original bounding box: {bndbox}')

    x0, x1 = bndbox[0] - offset_x, bndbox[1] + (width - crop_width - offset_x)
    y0, y1 = bndbox[2] - offset_y, bndbox[3] + (
            height - crop_height - offset_y)

    box = [x0, x1, y0, y1]

    box = shift_upper(box, max_x, max_y)
    box = shift_lower(box)
    log.debug(f'Updated bounding box: {box}')

    return crop_and_pad(image, box, width, height)


def center_crop_bbox(image: np.ndarray, bndbox: list, width: int = 299,
                     height: int = 299) -> torch.Tensor:
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
    y_max, x_max = image.shape[1], image.shape[2]

    crop_width = bndbox[1] - bndbox[0]
    crop_height = bndbox[3] - bndbox[2]

    x0, x1, y0, y1 = bndbox
    x0 -= np.floor((width - crop_width) / 2).astype(int)
    x1 += np.ceil((width - crop_width) / 2).astype(int)
    y0 -= np.floor((height - crop_height) / 2).astype(int)
    y1 += np.ceil((height - crop_height) / 2).astype(int)

    box = shift_upper([x0, x1, y0, y1], x_max, y_max)
    box = shift_lower(box)

    return crop_and_pad(image, box, width, height)


def crop_and_pad(image: np.ndarray, bndbox: List[int],
                 width: int = 299, height: int = 299) -> np.ndarray:
    """
    Crop image to specific size using bounding box,
    zero-pad if crop is too large.

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

    x0, x1, y0, y1 = bndbox

    image = image[y0:y1, x0:x1, :]

    image = np.pad(image,
                   ((0, height - image.shape[0]),
                    (0, width - image.shape[1]),
                    (0, 0)))

    return image
