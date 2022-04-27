import logging
from typing import Optional

import cv2 as cv
import numpy as np
from PyQt5.QtGui import QPen, QColorConstants as QColor, QBrush
from PyQt5.QtWidgets import QWidget, QMessageBox
from pyqtgraph import ImageItem, GraphicsView, ViewBox, PlotDataItem, TextItem

from ....data.common import Sample
from ...utils import run_in_main_thread

log = logging.getLogger(__name__)


class ImageWidget(GraphicsView):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.vb = ViewBox()
        self.vb.setAspectLocked()
        self.vb.invertY()
        self.setCentralItem(self.vb)

        self.item: ImageItem = ImageItem(axisOrder='row-major')
        self.box = PlotDataItem(pen=QPen(QColor.Black, 4), connect='all', )
        self.text = TextItem(color=QColor.Black, fill=QColor.White)

        self.pen_green = QPen(QColor.Green, 4)
        self.pen_red = QPen(QColor.Red, 4)

        self.vb.addItem(self.item)
        self.vb.addItem(self.box)
        self.vb.addItem(self.text)

    @run_in_main_thread
    def show_image(self, sample: Sample, evaluation: bool = False):
        try:
            image = cv.imread(sample.filepath)
        except OSError as e:
            QMessageBox.critical(
                self, 'Error',
                f'Error in loading image from {sample.filepath}\n{e}')
            return

        if not evaluation:
            box = np.array(sample.bbox.to_coords())
            label = sample.bbox.cls
            text = label
        else:
            if not sample.has_detection:
                raise ValueError(f'Sample {sample.eventId} has no detections.')

            if sample.num_detections > 1:
                log.warning(f'Sample {sample.eventId} has more than one '
                            f'detection. Don\'t know what to do.')
                return

            box = np.array(sample.detected_bbox[0].to_coords())
            label = sample.detected_bbox[0].cls
            score = sample.detected_bbox[0].score

            text = f'{label}: {score:.3f}'

        self.item.setImage(image)
        self.box.setData(box)
        self.text.setText(text)
        self.text.setPos(sample.bbox.x0, sample.bbox.y1)

        if label == 'DEFECT':
            self.box.setPen(self.pen_red)
            self.text.fill = QBrush(QColor.Red)
        else:
            self.box.setPen(self.pen_green)
            self.text.fill = QBrush(QColor.Green)

    def clear_plot(self):
        if self.item is None:
            return

        self.vb.removeItem(self.item)
