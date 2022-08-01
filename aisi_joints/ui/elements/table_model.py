from typing import Any
from typing import Optional

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel
from PyQt5.QtCore import QModelIndex
from PyQt5.QtCore import QObject
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ...data.common import Sample

HEADER = ["eventId", "sessionId", "platformId", "Label", "Split", "File path"]
HEADER_TO_COLUMN = {
    "eventId": "eventId",
    "platformId": "platformId",
    "sessionId": "sessionId",
    "Label": "cls",
    "Split": "split",
    "File path": "filepath",
}


class TableModel(QAbstractTableModel):
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        parent: Optional[QObject] = None,
    ):
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

        if "flagged" not in self._data.columns:
            self._data["flagged"] = False

        self._header = HEADER.copy()
        if "split" not in self._data.columns:
            self._header.remove("Split")
        if "sessionId" not in self._data.columns:
            self._header.remove("sessionId")

        self.modelReset.emit()

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[HEADER_TO_COLUMN[self._header[index.column()]]].iloc[
                index.row()
            ]
        elif role == Qt.ItemDataRole.BackgroundRole:
            if "detected_class" in self._data.columns:
                if (
                    self._data["cls"].iloc[index.row()]
                    != self._data["detected_class"].iloc[index.row()]
                ):
                    return QColor("yellow")
            if self._data["flagged"].iloc[index.row()]:
                return QColor("gray")
        elif role == Qt.ItemDataRole.ToolTipRole:
            if self._data["flagged"].iloc[index.row()]:
                return "This sample is flagged."

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self._header) if len(self._data) > 0 else 0

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = ...
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._header[section]
            elif orientation == Qt.Vertical:
                return f"{section}"

    def get_sample(self, row: int) -> Sample:
        return Sample.from_dataframe(self._data.iloc[row])

    def toggle_flagged(self, row: int):
        self._data.loc[self._data.index[row], "flagged"] = not self._data.loc[
            self._data.index[row], "flagged"
        ]

    def set_flagged(self, row: int):
        self._data["flagged"].iloc[row] = True

    def unset_flagged(self, row: int):
        self._data["flagged"].iloc[row] = False
