import logging
from pprint import pformat
from typing import Optional

import pandas as pd
from PyQt5.QtGui import QCloseEvent, QFont
from PyQt5.QtWidgets import QWidget
from sklearn.metrics import classification_report, confusion_matrix

from aisi_joints.constants import CLASS_OK, CLASS_DEFECT
from ...generated.metrics_widget_ui import Ui_MetricsWidget

log = logging.getLogger(__name__)


class MetricsWidget(Ui_MetricsWidget, QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        font = QFont()
        font.setFamily('monospace [Consolas]')
        font.setFixedPitch(True)
        font.setStyleHint(QFont.TypeWriter)

        self.textEdit.setFont(font)

    def show_metrics(self, df: pd.DataFrame):
        report = classification_report(
            df['cls'], df['detected_class'], digits=4
        )
        cf = confusion_matrix(df['cls'], df['detected_class'])

        msg = list()
        msg.append(('=' * 16) + ' CLASSIFICATION REPORT' + ('=' * 16))
        msg.append(report)
        msg.append('\n')
        msg.append(('=' * 16) + ' CONFUSION MATRIX ' + ('=' * 16))
        msg.append(pformat(cf))

        self.textEdit.setText('\n'.join(msg))

    def closeEvent(self, a0: QCloseEvent):
        if a0.spontaneous():
            self.hide()
            a0.ignore()
        else:
            a0.accept()
