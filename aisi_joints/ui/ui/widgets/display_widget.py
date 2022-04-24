import logging
from typing import Optional

import pandas as pd
from PyQt5.QtCore import QSortFilterProxyModel, QModelIndex, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox

from ..dialogs.export_dialog import ExportDialog
from ..dialogs.import_dialog import ImportDialog
from ...elements.table_model import TableModel
from ...settings import app

log = logging.getLogger(__name__)


class DisplayWidget(QWidget):
    data_loaded = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        table_model = TableModel(parent=self)
        proxy_model = QSortFilterProxyModel(self)
        proxy_model.setSourceModel(table_model)
        self.table_model = table_model
        self._proxy_model = proxy_model

        self._csv_path: str = None

    def show_current_img(self):
        raise NotImplementedError

    def show_img(self, index: QModelIndex):
        raise NotImplementedError

    @property
    def has_data(self) -> bool:
        return self._csv_path is not None

    def save(self):
        if self._csv_path is None:
            file, ok = QFileDialog.getSaveFileName(
                self, 'Select save file', app.current_dir, '*.csv')

            if not ok or file is None:
                log.debug('No file selected.')
                return

            app.current_dir = file
            self._csv_path = file

        try:
            log.info(f'Saving .csv to {self._csv_path}')
            self.table_model.dataframe.to_csv(self._csv_path, index=False)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))

    def load(self):
        file, ok = QFileDialog.getOpenFileName(
            self, 'Open .csv', app.current_dir, '*.csv')

        if not ok or file is None or file == '':
            log.debug('No file selected.')
            return

        app.current_dir = file

        df = pd.read_csv(file)
        self._csv_path = file

        self.table_model.dataframe = df

        self.data_loaded.emit()

    def import_(self):
        dialog = ImportDialog(self)

        def on_ok(df: pd.DataFrame, label_map: dict):
            self.table_model.dataframe = df

            self._csv_path = None

            QMessageBox.information(
                self, 'Import Success',
                f'Successfully imported {len(df)} samples')

            self.data_loaded.emit()

        dialog.files_imported.connect(on_ok)
        dialog.exec()

    def export_csv(self):
        dialog = ExportDialog(self.table_model.dataframe, parent=self)
        dialog.exec()
