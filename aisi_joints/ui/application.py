"""
Main application. Intended to be launched by __main__ entrypoint with
arguments.
"""

import logging
import sys
from argparse import Namespace

from PyQt5.QtWidgets import QApplication

from . import flags
from .ui.main_window import MainWindow
from .settings import app
from .. import __version__


def main(args: Namespace):
    # first initialise logger
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    if flags.DEBUG:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # second, initialize Qt application
    application = QApplication([])
    application.setApplicationVersion(__version__)
    application.setOrganizationName('SBB')
    application.setOrganizationDomain('sbb.ch')
    application.setApplicationName('AISI rail joints')

    # mute certain modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PyQt5.uic').setLevel(logging.WARNING)

    main_window = MainWindow()
    app.init()
    main_window.show()

    exit_code = application.exec_()
    sys.exit(exit_code)
