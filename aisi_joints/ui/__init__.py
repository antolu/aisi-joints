"""This module contains code for the joints viewer graphical application
written with for PyQt5."""


class Flags(object):
    def __init__(self, *items):
        for key, val in zip(items[:-1], items[1:]):
            setattr(self, key, val)


flags = Flags("DEBUG", False)


install_requires = [
    "PyQt5",
    "pyqtgraph",
    "pyqt5ac",
    "opencv-python" "pandas",
    "numpy",
]
