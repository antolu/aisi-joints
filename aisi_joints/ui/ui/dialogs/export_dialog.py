import logging
from os import path
from typing import Optional

import pandas as pd
from PyQt5.QtWidgets import QDialog, QWidget, QFileDialog, QMessageBox

from ...generated.export_dialog_ui import Ui_ExportDialog
from ...settings import app

log = logging.getLogger(__name__)


class ExportDialog(Ui_ExportDialog, QDialog):
    def __init__(self, to_export: pd.DataFrame, parent: Optional[QWidget] = None):
        QDialog.__init__(self, parent)
        self.setupUi(self)

        self._df = to_export

        self.buttonBrowse.clicked.connect(self.on_browse_clicked)
        self.buttonBox.accepted.connect(self.on_ok)

    def on_browse_clicked(self):
        current_dir = self.textBrowse.text()

        if current_dir == '':
            current_dir = app.current_dir
        else:
            current_dir = path.split(current_dir)[0]

        file, ok = QFileDialog.getSaveFileName(self, 'Export to csv',
                                               current_dir, '*.csv')

        if not ok or file is None or file == '':
            log.warning(f'Invalid path selected: {file}.')
            return

        app.current_dir = path.split(file)[0]
        self.textBrowse.setText(file)

    def on_ok(self):
        if self.textBrowse.text() == '':
            log.error('Must select export path first.')
            QMessageBox.critical(self, 'Error', 'Must select export path first.')

        df = self._df.copy()
        if not self.checkFlagged.isChecked():
            df = df[~df['flagged']]
        if not self.checkValidate.isChecked():
            df = df[~df['flagged']]

        log.info(f'Exporting .csv to {self.textBrowse.text()}.')

        try:
            df.to_csv(self.textBrowse.text(), index=False)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))
