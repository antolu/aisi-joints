import logging
from os import path
from typing import Optional

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox

from .display_widget import DisplayWidget
from ...elements.copy_action import CopySelectedCellsAction
from ...generated.sample_widget_ui import Ui_SampleWidget
from ...settings import app

log = logging.getLogger(__name__)


class SampleWidget(DisplayWidget, Ui_SampleWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        DisplayWidget.__init__(self, parent)
        self.setupUi(self)

        self.sampleTable.setModel(self._proxy_model)
        self.copy_action = CopySelectedCellsAction(self.sampleTable)

        self.sampleTable.doubleClicked.connect(self.show_img)
        self.sampleTable.activated.connect(self.show_img)

    def show_current_img(self):
        self.show_img(self.sampleTable.currentIndex())

    def show_img(self, index: QModelIndex):
        sample = self.table_model.get_sample(index.row())
        self.widget.show_image(sample)

    def export_flagged(self):
        df = self.table_model.dataframe
        file, ok = QFileDialog.getSaveFileName(self, 'Export flagged samples',
                                               app.current_dir, '*.csv')

        if not ok or file is None or file == '':
            if not ok or file is None or file == '':
                log.warning(f'Invalid path selected: {file}.')
                return

            app.current_dir = path.split(file)[0]

        log.info(f'Exporting flagged samples to {file}.')

        df = df[df['flaged']]

        try:
            df.to_csv(file, index=False)
        except OSError as e:
            QMessageBox.critical(self, 'Error', str(e))

    def flag_clicked(self):
        index = self.sampleTable.currentIndex()
        if index.row() == -1:
            return

        self.table_model.toggle_flagged(index.row())
        self.sampleTable.dataChanged(
            self.table_model.index(index.row(), 0),
            self.table_model.index(0, self.table_model.columnCount()))

    def validate_clicked(self):
        index = self.sampleTable.currentIndex()
        if index.row() == -1:
            return

        self.table_model.toggle_validate(index.row())
        self.sampleTable.dataChanged(
            self.table_model.index(index.row(), 0),
            self.table_model.index(0, self.table_model.columnCount()))
