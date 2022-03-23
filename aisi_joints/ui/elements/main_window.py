from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QMainWindow

import pandas as pd

from .table_model import TableModel
from ..generated.main_window_ui import Ui_MainWindow


class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self, *args, csv_path: str = None, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

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
        self.actionExit.triggered.connect(self.close)
        self.actionIgnore.triggered.connect(self.on_ignore_clicked)
        self.actionValidate.triggered.connect(self.on_validate_clicked)

    def on_double_click(self, index: QModelIndex):
        sample = self.table_model.get_sample(index.row())
        self.widget.show_image(sample)

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
