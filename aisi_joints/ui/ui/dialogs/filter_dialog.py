import logging
from typing import Optional

import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget, QFileDialog

from ...generated.filter_dialog_ui import Ui_FilterDialog
from ...settings import app
from ....data.filter_csv import filter_dataframe

log = logging.getLogger(__name__)


class FilterDialog(QDialog, Ui_FilterDialog):
    data_filtered = pyqtSignal(pd.DataFrame, str)

    def __init__(self, data: pd.DataFrame, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        self._data = data

        self.buttonBrowse.clicked.connect(self.on_browse)
        self.buttonBox.accepted.connect(self.on_ok)

    def on_browse(self):
        files, ok = QFileDialog.getOpenFileNames(
            self, 'Select filters', app.current_dir, '*.csv'
        )

        if not ok or len(files) == 0:
            log.debug('No files selected.')
            return

        app.current_dir = files[-1]

        self.textFiles.setText('\n'.join(files))

    def on_ok(self):
        files = self.textFiles.toPlainText().splitlines()

        filters = []
        for file in files:
            filters.append(pd.read_csv(file))

        df, msg = filter_dataframe(self._data.copy(), filters)

        self.data_filtered.emit(df, msg)
