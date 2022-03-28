from typing import Optional

import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget

from ..generated.partition_dialog_ui import Ui_PartitionDialog
import logging

from ...data.partition_dataset import partition_dataset

log = logging.getLogger(__name__)


class PartitionDialog(QDialog, Ui_PartitionDialog):
    data_partitioned = pyqtSignal(pd.DataFrame)

    def __init__(self, data: pd.DataFrame, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        self._data = data.copy()

        self.labelInfo.setText(f'Partition data with {len(data)} samples.')

        self.train_changed(self.spinBoxTrain.value())
        self.val_changed(self.spinBoxVal.value())
        self.test_changed(self.spinBoxTest.value())

        self.spinBoxTrain.valueChanged.connect(self.train_changed)
        self.spinBoxVal.valueChanged.connect(self.val_changed)
        self.spinBoxTest.valueChanged.connect(self.test_changed)

        self.buttonBox.accepted.connect(self.on_ok)

    def train_changed(self, value: int):
        total = value + self.spinBoxVal.value() + self.spinBoxTest.value()
        label = 'Train: {} samples'.format(round(value / total * len(self._data)))

        self.labelTrain.setText(label)

    def val_changed(self, value: int):
        total = value + self.spinBoxTrain.value() + self.spinBoxTest.value()
        label = 'Validation: {} samples'.format(round(value / total * len(self._data)))

        self.labelVal.setText(label)

    def test_changed(self, value: int):
        total = value + self.spinBoxTrain.value() + self.spinBoxVal.value()
        label = 'Test: {} samples'.format(round(value / total * len(self._data)))

        self.labelTest.setText(label)

    def on_ok(self):
        ratio = '{}/{}/{}'.format(self.spinBoxTrain.value(),
                                  self.spinBoxVal.value(),
                                  self.spinBoxTest.value())

        df = partition_dataset(self._data, ratio)

        self.data_partitioned.emit(df)
