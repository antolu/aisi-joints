import logging
from typing import Optional

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QProgressDialog

from .dialogs.filter_dialog import FilterDialog
from .dialogs.partition_dialog import PartitionDialog
from .dialogs.update_filepaths_dialog import UpdateFilepathsDialog
from ..elements.table_model import TableModel
from ..settings import app
from ..utils import run_in_main_thread
from ...constants import LABEL_MAP
from ...data.common import write_pbtxt
from ...data.generate_tfrecord import generate_tfrecord

log = logging.getLogger(__name__)


class ToolsMenu:
    def __init__(self, table_model: TableModel, parent: Optional[QWidget] = None):
        self._parent = parent
        self._table_model = table_model

    def on_partition(self):
        dialog = PartitionDialog(self._table_model.dataframe, self._parent)

        def on_ok(df: pd.DataFrame):
            self._table_model.dataframe = df

            QMessageBox.information(
                self._parent, 'Partition success', 'Successfully partitioned dataset.')
            # TODO: add more partition information

        dialog.data_partitioned.connect(on_ok)
        dialog.exec()

    def on_filter(self):
        dialog = FilterDialog(self._table_model.dataframe, self._parent)

        def on_ok(df: pd.DataFrame):
            self._table_model.dataframe = df

            QMessageBox.information(
                self._parent, 'Filter success', 'Successfully filtered dataset.')

        dialog.data_filtered.connect(on_ok)
        dialog.exec()

    def on_update_paths(self):
        dialog = UpdateFilepathsDialog(self._table_model.dataframe, self._parent)

        def on_ok(df: pd.DataFrame):
            self._table_model.dataframe = df

            QMessageBox.information(
                self._parent, 'Update file paths success.',
                f'Successfully updated file paths for {len(df)} samples.')

        dialog.paths_updated.connect(on_ok)
        dialog.exec()

    def on_generate_tfrecord(self):
        directory = QFileDialog.getExistingDirectory(
            self._parent, 'Select .tfrecord output directory.', app.current_dir)

        if directory is None or directory == '':
            return

        app.current_dir = directory

        df = self._table_model.dataframe
        progress_window = QProgressDialog(
            'Generating .tfrecords...', 'Cancel', 0, len(df), self._parent)
        progress_window.setWindowModality(Qt.WindowModal)
        progress_window.show()

        @run_in_main_thread
        def progress_callback(value: int, _):
            progress_window.setValue(value)

        try:
            msg = generate_tfrecord(
                self._table_model.dataframe, LABEL_MAP, directory,
                use_class_weights=True, progress_cb=progress_callback)
            error_occurred = False
        except OSError as e:
            error_occurred = True
            QMessageBox.critical(
                self._parent, 'Error',
                f'Unknown error while generating tfrecords: \n{str}')
        finally:
            progress_window.close()

        if error_occurred:
            return

        QMessageBox.information(self._parent, 'Success!', msg)

    def on_export_labelmap(self):
        file, ok = QFileDialog.getSaveFileName(
            self._parent, 'Export labelmap', app.current_dir, '*.pbtxt')

        if not ok or file is None or file == '':
            return

        app.current_dir = file

        write_pbtxt(LABEL_MAP, file)
