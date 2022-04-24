import logging

from PyQt5.QtWidgets import QMainWindow, QActionGroup

from .tools_menu import ToolsMenu
from ..generated.main_window_ui import Ui_MainWindow

log = logging.getLogger(__name__)


class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.tools_menu = ToolsMenu(self.sampleWidget.table_model, self)

        self.actionShow_Image.triggered.connect(
            lambda: self.dispatch_action('show_current_img'))
        self.actionImport.triggered.connect(
            lambda: self.dispatch_action('import_'))
        self.actionLoad_csv.triggered.connect(
            lambda: self.dispatch_action('load'))
        self.actionSave_csv.triggered.connect(
            lambda: self.dispatch_action('save'))
        self.actionExport_csv.triggered.connect(
            lambda: self.dispatch_action('export_csv'))

        # sample widget actions
        self.actionIgnore.triggered.connect(
            self.sampleWidget.ignore_clicked)
        self.actionValidate.triggered.connect(
            self.sampleWidget.validate_clicked)
        self.actionExportIgnored.triggered.connect(
            self.sampleWidget.export_ignored)
        self.action_ExportRevalidation.triggered.connect(
            self.sampleWidget.export_revalidation)

        # manipulate training data actions
        self.action_Partition_Dataset.triggered.connect(
            self.tools_menu.on_partition)
        self.action_Filter_Dataset.triggered.connect(
            self.tools_menu.on_filter)
        self.action_Update_Filepaths.triggered.connect(
            self.tools_menu.on_update_paths)
        self.action_Generate_tfrecord.triggered.connect(
            self.tools_menu.on_generate_tfrecord)
        self.actionExport_Labelmap.triggered.connect(
            self.tools_menu.on_export_labelmap)

        self.actionExit.triggered.connect(self.close)

        self.modeActionGroup = QActionGroup(self)
        self.modeActionGroup.addAction(self.actionTrain)
        self.modeActionGroup.addAction(self.actionEvaluation)
        self.modeActionGroup.setExclusive(True)
        self.actionTrain.changed.connect(self.change_mode)
        self.actionEvaluation.changed.connect(self.change_mode)

        self.sampleWidget.data_loaded.connect(self.update_action_state)
        self.evalWidget.data_loaded.connect(self.update_action_state)

        self.disable_data_actions()
        self.disable_sample_actions()

    def dispatch_action(self, action_name: str):
        getattr(self.stackedWidget.currentWidget(), action_name)()

    def change_mode(self):
        if self.actionTrain.isChecked():
            self.stackedWidget.setCurrentWidget(self.sampleWidget)
        else:
            self.stackedWidget.setCurrentWidget(self.evalWidget)

        self.update_action_state()

    def update_action_state(self):
        if self.stackedWidget.currentWidget() is self.sampleWidget:
            self.enable_sample_actions()
        else:
            self.disable_sample_actions()

        self.enable_data_actions()

    def enable_data_actions(self):
        self.actionSave_csv.setEnabled(True)
        self.actionShow_Image.setEnabled(True)
        self.actionExport_csv.setEnabled(True)

    def disable_data_actions(self):
        self.actionSave_csv.setDisabled(True)
        self.actionShow_Image.setDisabled(True)
        self.actionExport_csv.setDisabled(True)

    def enable_sample_actions(self):
        self.action_Partition_Dataset.setEnabled(True)
        self.action_Filter_Dataset.setEnabled(True)
        self.action_Update_Filepaths.setEnabled(True)
        self.action_Generate_tfrecord.setEnabled(True)
        self.actionExport_Labelmap.setEnabled(True)

        self.actionIgnore.setEnabled(True)
        self.actionValidate.setEnabled(True)

        self.actionExportIgnored.setEnabled(True)
        self.action_ExportRevalidation.setEnabled(True)

    def disable_sample_actions(self):
        self.action_Partition_Dataset.setDisabled(True)
        self.action_Filter_Dataset.setDisabled(True)
        self.action_Update_Filepaths.setDisabled(True)
        self.action_Generate_tfrecord.setDisabled(True)
        self.actionExport_Labelmap.setDisabled(True)

        self.actionIgnore.setDisabled(True)
        self.actionValidate.setDisabled(True)

        self.actionExportIgnored.setDisabled(True)
        self.action_ExportRevalidation.setDisabled(True)
