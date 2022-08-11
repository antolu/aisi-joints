import logging
from os import path
from typing import Optional, List

import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
    QFileDialog,
    QListView,
    QTreeView,
    QFileSystemModel,
    QAbstractItemView,
    QMessageBox,
)

from ...generated.import_dialog_ui import Ui_ImportDialog
from ...settings import app
from ....data.import_pascal_voc import import_pascal_voc
from ....data.import_rcm_api import import_rcm_api

log = logging.getLogger(__name__)


class ImportDialog(QDialog, Ui_ImportDialog):
    files_imported = pyqtSignal(pd.DataFrame, dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        self.buttonBrowseLabel.clicked.connect(self.browse_label)
        self.buttonBrowseBox.clicked.connect(self.browse_box)
        self.buttonBrowseImg.clicked.connect(self.browse_img)

        self.buttonBox.accepted.connect(self.on_ok)

        self.radioRcm.toggled.connect(self.import_type_changed)
        self.radioPascal.toggled.connect(self.import_type_changed)

    def import_type_changed(self, _: bool):
        if self.radioRcm.isChecked():
            self.labelLabel.setText('CSV with labels')
            self.labelBox.setText('CSV with boxes')
            self.checkDeviation.setHidden(False)
        elif self.radioPascal.isChecked():
            self.labelLabel.setText('CSV with RCM metadata')
            self.labelBox.setText('Diretories containing label XML files.')
            self.checkDeviation.setHidden(True)

    def browse_label(self):
        files, ok = QFileDialog.getOpenFileNames(
            self, 'Select .csv files with labels', app.current_dir, '*.csv'
        )

        if not ok or len(files) == 0:
            log.debug('No files selected.')
            return

        app.current_dir = files[-1]
        self.textLabel.setText('\n'.join(files))

    def browse_box(self):
        if self.radioRcm.isChecked():
            files, ok = QFileDialog.getOpenFileNames(
                self, 'Select .csv files with boxes', app.current_dir, '*.csv'
            )

            if not ok or len(files) == 0:
                log.debug('No files selected.')
                return

            app.current_dir = files[-1]
            self.textBox.setText('\n'.join(files))
        elif self.radioPascal.isChecked():
            directories = self._choose_directories()

            if len(directories) == 0:
                log.debug('No directories selected.')
                return

            app.current_dir = path.split(directories[-1])[0]
            self.textBox.setText('\n'.join(directories))

    def browse_img(self):
        directories = self._choose_directories()

        if len(directories) == 0:
            log.debug('No directories selected.')
            return

        app.current_dir = path.split(directories[-1])[0]
        self.textImg.setText('\n'.join(directories))

    def on_ok(self):
        labels = self.textLabel.toPlainText().split('\n')
        boxes = self.textBox.toPlainText().split('\n')
        images = self.textImg.toPlainText().split('\n')
        deviations_only = self.checkDeviation.isChecked()

        if self.radioRcm.isChecked():
            try:
                df, label_map = import_rcm_api(
                    labels, boxes, images, deviations_only
                )
            except (KeyError, OSError) as e:
                log.exception(
                    'An error occurred while importing RCM API .csv files.'
                )
                QMessageBox.critical(
                    self,
                    'Error',
                    'An error occurred while importing '
                    'RCM API .csv files: \n' + str(e),
                )
                return
        elif self.radioPascal.isChecked():
            try:
                df, label_map = import_pascal_voc(labels, boxes, images)
            except (KeyError, OSError) as e:
                log.exception(
                    'An error occurred while importing PASCAL VOC format datasets.'
                )
                QMessageBox.critical(
                    self,
                    'Error',
                    'An error occurred while importing '
                    'PASCAL VOC format datasets: \n' + str(e),
                )
                return
        else:
            return

        self.files_imported.emit(df, label_map)

    def _choose_directories(self) -> List[str]:
        dialog = QFileDialog(self)
        dialog.setWindowTitle('Select directories with images')
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        for view in dialog.findChildren((QListView, QTreeView)):
            if isinstance(view.model(), QFileSystemModel):
                view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        dialog.deleteLater()

        if dialog.exec() == QDialog.Accepted:
            return dialog.selectedFiles()
        else:
            return []
