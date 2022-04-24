import logging
from typing import Optional

from PyQt5.QtWidgets import QWidget

from .display_widget import DisplayWidget
from ...generated.eval_widget_ui import Ui_EvalWidget

log = logging.getLogger(__name__)


class EvalWidget(DisplayWidget, Ui_EvalWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)
