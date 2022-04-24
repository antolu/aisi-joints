import logging
from typing import Optional

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QWidget, QMessageBox

from .display_widget import DisplayWidget
from ...elements.copy_action import CopySelectedCellsAction
from ...generated.eval_widget_ui import Ui_EvalWidget

log = logging.getLogger(__name__)


class EvalWidget(DisplayWidget, Ui_EvalWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        self.sampleTable.setModel(self._proxy_model)
        self.copy_action = CopySelectedCellsAction(self.sampleTable)

        self.sampleTable.doubleClicked.connect(self.show_img)
        self.sampleTable.activated.connect(self.show_img)

    def show_current_img(self):
        self.show_img(self.sampleTable.currentIndex())

    def show_img(self, index: QModelIndex):
        sample = self.table_model.get_sample(index.row())

        try:
            self.widgetGt.show_image(sample)
            self.widgetPred.show_image(sample, evaluation=True)
        except ValueError as e:
            log.exception('An exception occurred while showing image.')
            QMessageBox.critical(self, 'Error', 'An exception occurred: '
                                 + str(e))
            return
