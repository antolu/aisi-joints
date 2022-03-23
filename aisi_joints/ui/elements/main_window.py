from os import path

from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox

import pandas as pd

from .export_dialog import ExportDialog
from .table_model import TableModel
from ..generated.main_window_ui import Ui_MainWindow
from ..settings import app


import logging

log = logging.getLogger(__name__)

class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self, *args, csv_path: str = None, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.csv_path = csv_path
        if csv_path is None:
            df = pd.DataFrame()
        else:
            df = pd.read_csv(csv_path)

        table_model = TableModel(data=df, parent=self)
        proxy_model = QSortFilterProxyModel(self)
        proxy_model.setSourceModel(table_model)
        self.sampleTable.setModel(proxy_model)
        self.table_model = table_model

        self.sampleTable.doubleClicked.connect(self.on_double_click)
        self.sampleTable.activated.connect(self.on_double_click)
        self.actionShow_Image.triggered.connect(self.on_double_click)
        self.actionSave_csv.triggered.connect(self.on_save)
        self.actionExit.triggered.connect(self.close)
        self.actionIgnore.triggered.connect(self.on_ignore_clicked)
        self.actionValidate.triggered.connect(self.on_validate_clicked)
        self.actionExport_csv.triggered.connect(self.on_export_csv)
        self.actionExportIgnored.triggered.connect(self.on_export_ignored)
        self.action_ExportRevalidation.triggered.connect(self.on_export_revalidation)

    def on_double_click(self, index: QModelIndex):
        sample = self.table_model.get_sample(index.row())
        self.widget.show_image(sample)

    def on_save(self):
        log.info(f'Saving .csv to {self.csv_path}')
        try:
            self.table_model.dataframe.to_csv(self.csv_path)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))

    def on_export_csv(self):
        dialog = ExportDialog(self.table_model.dataframe, parent=self)
        dialog.exec()

    def on_export_ignored(self):
        df = self.table_model.dataframe
        file, ok = QFileDialog.getSaveFileName(self, 'Export ignored samples',
                                               app.current_dir, '*.csv')

        if not ok or file is None or file == '':
            if not ok or file is None or file == '':
                log.warning(f'Invalid path selected: {file}.')
                return

            app.current_dir = path.split(file)[0]

        log.info(f'Exporting ignored samples to {file}.')

        df = df[df['ignore']]

        try:
            df.to_csv(file, index=False)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))

    def on_export_revalidation(self):
        df = self.table_model.dataframe
        file, ok = QFileDialog.getSaveFileName(self, 'Export revalidation samples',
                                               app.current_dir, '*.csv')

        if not ok or file is None or file == '':
            if not ok or file is None or file == '':
                log.warning(f'Invalid path selected: {file}.')
                return

            app.current_dir = path.split(file)[0]

        log.info(f'Exporting revalidation samples to {file}.')

        df = df[df['validate']]

        try:
            df.to_csv(file, index=False)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))

    def on_ignore_clicked(self):
        index = self.sampleTable.currentIndex()

        self.table_model.toggle_ignore(index.row())
        self.sampleTable.dataChanged(
            self.table_model.index(index.row(), 0),
            self.table_model.index(0, self.table_model.columnCount()))

    def on_validate_clicked(self):
        index = self.sampleTable.currentIndex()

        self.table_model.toggle_validate(index.row())
        self.sampleTable.dataChanged(
            self.table_model.index(index.row(), 0),
            self.table_model.index(0, self.table_model.columnCount()))
