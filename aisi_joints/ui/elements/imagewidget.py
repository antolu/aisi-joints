from typing import Optional, Tuple

from PyQt5.QtWidgets import QWidget
from pyqtgraph import GraphicsLayoutWidget, PlotItem, ImageItem, GraphicsView, ViewBox
import numpy as np
import pandas as pd
import cv2 as cv

from ..utils import run_in_main_thread


class ImageWidget(GraphicsView):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.vb = ViewBox()
        self.vb.setAspectLocked()
        self.vb.invertY()
        self.setCentralItem(self.vb)

        self.item: ImageItem = None

    @run_in_main_thread
    def show_image(self, sample: pd.DataFrame):
        image = cv.imread(sample.filepath)

        if self.item is None:
            self.item = ImageItem(image, axisOrder='row-major')
            self.vb.addItem(self.item)
        else:
            self.item.setImage(image)

    def clear_plot(self):
        if self.item is None:
            return

        self.vb.removeItem(self.item)
