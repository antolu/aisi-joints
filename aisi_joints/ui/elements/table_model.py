from typing import Optional, Any

from PyQt5.QtCore import QAbstractTableModel, QObject, QModelIndex, Qt
import pandas as pd
import numpy as np
from PyQt5.QtGui import QColor

HEADER = ['eventId', 'sessionId', 'platformId', 'Label', 'Split', 'File path']
HEADER_TO_COLUMN = {
    'eventId': 'eventId',
    'platformId': 'platformId',
    'sessionId': 'sessionId',
    'Label': 'cls',
    'Split': 'split',
    'File path': 'filepath'
}


class TableModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, parent: Optional[QObject] = None):
        super().__init__(parent)

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

    def get_sample(self, row: int) -> pd.DataFrame:
        return self._data.iloc[row]

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
