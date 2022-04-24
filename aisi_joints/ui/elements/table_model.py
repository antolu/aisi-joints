from typing import Optional, Any, List

from PyQt5.QtCore import QAbstractTableModel, QObject, QModelIndex, Qt
import pandas as pd
import numpy as np
from PyQt5.QtGui import QColor
from dataclasses import dataclass

HEADER = ['eventId', 'sessionId', 'platformId', 'Label', 'Split', 'File path']
HEADER_TO_COLUMN = {
    'eventId': 'eventId',
    'platformId': 'platformId',
    'sessionId': 'sessionId',
    'Label': 'cls',
    'Split': 'split',
    'File path': 'filepath'
}


@dataclass
class DetectionBox:
    x0: int
    x1: int
    y0: int
    y1: int
    cls: str
    score: float = -1

    def to_coords(self) -> List[List[int]]:
        return [[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1],
                [self.x0, self.y1], [self.x0, self.y0]]


class Sample:
    eventId: str
    filepath: str
    bbox: DetectionBox

    has_detection: bool
    num_detections: int

    detected_bbox = List[DetectionBox]

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> 'Sample':
        sample = Sample()
        sample.eventId = df['eventId']
        sample.filepath = df['filepath']
        sample.bbox = DetectionBox(df['x0'], df['x1'], df['y0'], df['y1'],
                                   df['cls'])

        if 'detected_class' not in df.index:
            sample.has_detection = False
            return sample

        sample.has_detection = True
        sample.num_detections = df['num_detections']

        sample.detected_bbox = []
        x0 = [o for o in map(int, str(df['detected_x0']).split(';'))]
        x1 = [o for o in map(int, str(df['detected_x1']).split(';'))]
        y0 = [o for o in map(int, str(df['detected_y0']).split(';'))]
        y1 = [o for o in map(int, str(df['detected_y1']).split(';'))]
        scores = [o for o in map(float, str(df['detection_score']).split(';'))]
        cls = [o for o in df['detected_class'].split(';')]

        for i in range(sample.num_detections):
            sample.detected_bbox.append(
                DetectionBox(x0[i], x1[i], y0[i], y1[i], cls[i], scores[i]))

        return sample


class TableModel(QAbstractTableModel):
    def __init__(self, data: Optional[pd.DataFrame] = None,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if data is None:
            data = pd.DataFrame()

        self.dataframe = data

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._data.copy()

    @dataframe.setter
    def dataframe(self, new: pd.DataFrame):
        self._data = new.copy()

        if 'ignore' not in self._data.columns:
            self._data['ignore'] = False

        if 'validate' not in self._data.columns:
            self._data['validate'] = False

        self._header = HEADER.copy()
        if 'split' not in self._data.columns:
            self._header.remove('Split')
        if 'sessionId' not in self._data.columns:
            self._header.remove('sessionId')

        self.modelReset.emit()

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[HEADER_TO_COLUMN[self._header[index.column()]]].iloc[index.row()]
        elif role == Qt.ItemDataRole.BackgroundRole:
            if self._data['ignore'].iloc[index.row()]:
                return QColor('gray')
            elif self._data['validate'].iloc[index.row()]:
                return QColor('red')
        elif role == Qt.ItemDataRole.ToolTipRole:
            if self._data['ignore'].iloc[index.row()]:
                return 'This sample is marked as ignored.'
            elif self._data['validate'].iloc[index.row()]:
                return 'This sample is marked as needing revalidation.'

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self._header) if len(self._data) > 0 else 0

    def headerData(self, section: int,
                   orientation: Qt.Orientation, role: int = ...) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._header[section]
            elif orientation == Qt.Vertical:
                return f'{section}'

    def get_sample(self, row: int) -> Sample:
        return Sample.from_dataframe(self._data.iloc[row])

    def toggle_ignore(self, row: int):
        self._data.loc[self._data.index[row], 'ignore'] = \
            not self._data.loc[self._data.index[row], 'ignore']

    def set_ignore(self, row: int):
        self._data['ignore'].iloc[row] = True

    def unset_ignore(self, row: int):
        self._data['ignore'].iloc[row] = False

    def toggle_validate(self, row: int):
        self._data.loc[self._data.index[row], 'validate'] = \
            not self._data.loc[self._data.index[row], 'validate']

    def set_validate(self, row: int):
        self._data['validate'].iloc[row] = True

    def unset_validate(self, row: int):
        self._data['validate'].iloc[row] = False
