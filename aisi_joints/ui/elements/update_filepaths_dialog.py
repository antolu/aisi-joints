import logging
from os import path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget, QFileDialog
import pandas as pd

from ..generated.update_filepaths_dialog_ui import Ui_UpdateFilepathsDialog
from ..settings import app
from ..utils import choose_directories
from ...data.update_filepaths import update_paths

log = logging.getLogger(__name__)


class UpdateFilepathsDialog(QDialog, Ui_UpdateFilepathsDialog):
    paths_updated = pyqtSignal(pd.DataFrame)

    def __init__(self, data: pd.DataFrame, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        self._data = data

        self.buttonBrowse.clicked.connect(self.on_browse)
        self.buttonBox.accepted.connect(self.on_ok)

    def on_browse(self):
        directories = choose_directories(self)

        if len(directories) == 0:
            log.debug('No directories selected.')
            return
        log.debug(f'Selected directories: {", ".join(directories)}.')

        app.current_dir = path.split(directories[-1])[0]
        self.textFiles.setText('\n'.join(directories))

    def on_ok(self):
        directories = self.textFiles.toPlainText().splitlines()

        df = update_paths(self._data.copy(), directories)

        self.paths_updated.emit(df)