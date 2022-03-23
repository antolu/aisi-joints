from typing import Optional, Any

from PyQt5.QtCore import QAbstractTableModel, QObject, QModelIndex, Qt
import pandas as pd
import numpy as np


HEADER = ['eventId', 'platformId', 'Label', 'Split', 'File path']
HEADER_TO_COLUMN = {
    'eventId': 'eventId',
    'platformId': 'platformId',
    'Label': 'cls',
    'Split': 'split',
    'File path': 'filepath'
}


class TableModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._data = data

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[HEADER_TO_COLUMN[HEADER[index.column()]]].iloc[index.row()]

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(HEADER) if len(self._data) > 0 else 0

    def headerData(self, section: int,
                   orientation: Qt.Orientation, role: int = ...) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Horizontal:
                return HEADER[section]
            elif orientation == Qt.Vertical:
                return f'{section}'

    def get_sample(self, row: int) -> pd.DataFrame:
        return self._data.iloc[row]
