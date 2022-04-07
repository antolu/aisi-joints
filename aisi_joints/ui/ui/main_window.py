import logging
from os import path

import pandas as pd
from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QProgressDialog

from .dialogs.export_dialog import ExportDialog
from .dialogs.filter_dialog import FilterDialog
from .dialogs.import_dialog import ImportDialog
from .dialogs.partition_dialog import PartitionDialog
from .dialogs.update_filepaths_dialog import UpdateFilepathsDialog
from .tools_menu import ToolsMenu
from ..elements.copy_action import CopySelectedCellsAction
from ..elements.table_model import TableModel
from ..generated.main_window_ui import Ui_MainWindow
from ..settings import app
from ..utils import run_in_main_thread
from ...constants import LABEL_MAP
from ...data.common import write_pbtxt
from ...data.generate_tfrecord import generate_tfrecord

log = logging.getLogger(__name__)


class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self, *args, csv_path: str = None, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.csv_path = csv_path
        if csv_path is None:
            df = pd.DataFrame()
            self.disable_data_actions()
        else:
            df = pd.read_csv(csv_path)
            self.enable_data_actions()

        table_model = TableModel(data=df, parent=self)
        proxy_model = QSortFilterProxyModel(self)
        proxy_model.setSourceModel(table_model)
        self.sampleTable.setModel(proxy_model)
        self.table_model = table_model

        self.tools_menu = ToolsMenu(self.table_model, self)

        self.copy_action = CopySelectedCellsAction(self.sampleTable)

        self.sampleTable.doubleClicked.connect(self.on_double_click)
        self.sampleTable.activated.connect(self.on_double_click)
        self.actionShow_Image.triggered.connect(
            lambda: self.on_double_click(self.sampleTable.currentIndex()))
        self.actionImport.triggered.connect(self.on_import)
        self.actionLoad_csv.triggered.connect(self.on_load)
        self.actionSave_csv.triggered.connect(self.on_save)
        self.actionExit.triggered.connect(self.close)
        self.actionIgnore.triggered.connect(self.on_ignore_clicked)
        self.actionValidate.triggered.connect(self.on_validate_clicked)
        self.actionExport_csv.triggered.connect(self.on_export_csv)
        self.actionExportIgnored.triggered.connect(self.on_export_ignored)
        self.action_ExportRevalidation.triggered.connect(self.on_export_revalidation)

        self.action_Partition_Dataset.triggered.connect(self.tools_menu.on_partition)
        self.action_Filter_Dataset.triggered.connect(self.tools_menu.on_filter)
        self.action_Update_Filepaths.triggered.connect(self.tools_menu.on_update_paths)
        self.action_Generate_tfrecord.triggered.connect(self.tools_menu.on_generate_tfrecord)
        self.actionExport_Labelmap.triggered.connect(self.tools_menu.on_export_labelmap)

    def on_double_click(self, index: QModelIndex):
        sample = self.table_model.get_sample(index.row())
        self.widget.show_image(sample)

    def on_save(self):
        if self.csv_path is None:
            file, ok = QFileDialog.getSaveFileName(
                self, 'Select save file', app.current_dir, '*.csv')

            if not ok or file is None:
                log.debug('No file selected.')
                return

            app.current_dir = file
            self.csv_path = file

        try:
            log.info(f'Saving .csv to {self.csv_path}')
            self.table_model.dataframe.to_csv(self.csv_path)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))

    def on_load(self):
        file, ok = QFileDialog.getOpenFileName(
            self, 'Open .csv', app.current_dir, '*.csv')

        if not ok or file is None or file == '':
            log.debug('No file selected.')
            return

        app.current_dir = file

        df = pd.read_csv(file)
        self.csv_path = file

        self.table_model.dataframe = df

        self.enable_data_actions()

    def on_import(self):
        dialog = ImportDialog(self)

        def on_ok(df: pd.DataFrame, label_map: dict):
            self.table_model.dataframe = df

            self.csv_path = None

            QMessageBox.information(
                self, 'Import Success',
                f'Successfully imported {len(df)} samples')

            self.enable_data_actions()

        dialog.files_imported.connect(on_ok)
        dialog.exec()

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
        if index.row() == -1:
            return

        self.table_model.toggle_ignore(index.row())
        self.sampleTable.dataChanged(
            self.table_model.index(index.row(), 0),
            self.table_model.index(0, self.table_model.columnCount()))

    def on_validate_clicked(self):
        index = self.sampleTable.currentIndex()
        if index.row() == -1:
            return

        self.table_model.toggle_validate(index.row())
        self.sampleTable.dataChanged(
            self.table_model.index(index.row(), 0),
            self.table_model.index(0, self.table_model.columnCount()))

    def enable_data_actions(self):
        self.actionSave_csv.setEnabled(True)
        self.action_Partition_Dataset.setEnabled(True)
        self.action_Filter_Dataset.setEnabled(True)
        self.action_Update_Filepaths.setEnabled(True)
        self.action_Generate_tfrecord.setEnabled(True)
        # self.actionExport_Labelmap.setEnabled(True)
        self.actionShow_Image.setEnabled(True)
        self.actionIgnore.setEnabled(True)
        self.actionValidate.setEnabled(True)
        self.actionExport_csv.setEnabled(True)
        self.actionExportIgnored.setEnabled(True)
        self.action_ExportRevalidation.setEnabled(True)

    def disable_data_actions(self):
        self.actionSave_csv.setDisabled(True)
        self.action_Partition_Dataset.setDisabled(True)
        self.action_Filter_Dataset.setDisabled(True)
        self.action_Update_Filepaths.setDisabled(True)
        self.action_Generate_tfrecord.setDisabled(True)
        # self.actionExport_Labelmap.setDisabled(True)
        self.actionShow_Image.setDisabled(True)
        self.actionIgnore.setDisabled(True)
        self.actionValidate.setDisabled(True)
        self.actionExport_csv.setDisabled(True)
        self.actionExportIgnored.setDisabled(True)
        self.action_ExportRevalidation.setDisabled(True)