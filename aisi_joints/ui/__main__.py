#!/usr/bin/env python

"""
Main entrypoint of the application. Launch the application as
`python -m atom_fixdisplay`.
"""
import os.path as path
from argparse import ArgumentParser

import pyqt5ac

from . import flags
from .application import main

parser = ArgumentParser()
parser.add_argument('-d', '--debug', '--develop', dest='debug',
                    action='store_true',
                    help='Enable debug mode. Debug level log messages will '
                         'be shown in logging output.')
parser.add_argument('-c', '--csv', type=str, default=None, help='Name of split .csv file.')

args = parser.parse_args()
flags.DEBUG = args.debug

# if in debug mode, generate pyqt views from ui xml files
if flags.DEBUG:
    try:
        # compile pyqt files
        HERE = path.split(path.abspath(__file__))[0]
        pyqt5ac.main(
             uicOptions='--from-imports', force=False, initPackage=True,
            ioPaths=[
                [path.join(HERE, '../resources/ui/*.ui'),
                 path.join(HERE, 'generated/%%FILENAME%%_ui.py')],
                [path.join(HERE, 'resources/*.qrc'),
                 path.join(HERE, 'aisi_joints/ui/generated/%%FILENAME%%_rc.py')]
            ]
        )
    except PermissionError:
        pass

main(args)
