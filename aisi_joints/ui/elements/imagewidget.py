from typing import Optional

import cv2 as cv
import numpy as np
import pandas as pd
from PyQt5.QtGui import QPen, QColorConstants as QColor
from PyQt5.QtWidgets import QWidget, QMessageBox
from pyqtgraph import ImageItem, GraphicsView, ViewBox, PlotDataItem

from ..utils import run_in_main_thread


class ImageWidget(GraphicsView):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.vb = ViewBox()
        self.vb.setAspectLocked()
        self.vb.invertY()
        self.setCentralItem(self.vb)

        self.item: ImageItem = ImageItem(axisOrder='row-major')
        self.box = PlotDataItem(pen=QPen(QColor.Black, 4), connect='all', )

        self.pen_green = QPen(QColor.Green, 4)
        self.pen_red = QPen(QColor.Red, 4)

        self.vb.addItem(self.item)
        self.vb.addItem(self.box)

    @run_in_main_thread
    def show_image(self, sample: pd.DataFrame):
        try:
            image = cv.imread(sample.filepath)
        except OSError as e:
            QMessageBox.critical(
                self, 'Error',
                f'Error in loading image from {sample.filepath}\n{e}')
            return

        x0 = sample.x0
        x1 = sample.x1
        y0 = sample.y0
        y1 = sample.y1
        label = sample.cls

        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])

        self.item.setImage(image)
        self.box.setData(box)

        if label == 'DEFECT':
            self.box.setPen(self.pen_red)
        else:
            self.box.setPen(self.pen_green)

    def clear_plot(self):
        if self.item is None:
            return

        self.vb.removeItem(self.item)
