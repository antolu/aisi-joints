import logging

from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QTableView

log = logging.getLogger(__name__)


class CopySelectedCellsAction(QAction):
    def __init__(self, table_widget):
        if not isinstance(table_widget, QTableView):
            raise ValueError(
                f"CopySelectedCellsAction must be initialised with a QTableView. A {type(table_widget)} was given."
            )
        super().__init__("Copy", table_widget)
        self.setShortcut("Ctrl+C")
        self.triggered.connect(self.copy_cells_to_clipboard)
        self.table = table_widget

    def copy_cells_to_clipboard(self):
        if len(self.table.selectionModel().selectedIndexes()) > 0:
            # sort select indexes into rows and columns
            previous = self.table.selectionModel().selectedIndexes()[0]
            columns = []
            rows = []
            for index in self.table.selectionModel().selectedIndexes():
                if previous.column() != index.column():
                    columns.append(rows)
                    rows = []
                rows.append(index.data())
                previous = index
            columns.append(rows)

            # add rows and columns to clipboard
            clipboard = ""
            nrows = len(columns[0])
            ncols = len(columns)
            for r in range(nrows):
                for c in range(ncols):
                    clipboard += columns[c][r]
                    if c != (ncols - 1):
                        clipboard += "\t"
                clipboard += "\n"

            # copy to the system clipboard
            sys_clip = QApplication.clipboard()
            sys_clip.setText(clipboard)

            log.debug(f"Copied to clipboard: {clipboard}")
