import logging
import os
import time as t
from os import path
from typing import Callable

__all__ = ['time_execution', 'get_latest']

log = logging.getLogger(__name__)


class time_execution:
    """
    Convenience class for timing execution. Used simply as
    >>> with time_execution() as t:
    >>>     # some code to time
    >>> print(t.duration)
    """

    def __init__(self):
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        self.start = t.time()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = t.time()
        self.duration = self.end - self.start


def get_latest(dir_, condition: Callable):
    if path.isdir(dir_):
        files = [path.join(dir_, o) for o in os.listdir(dir_) if condition(o)]

        latest = max(files, key=path.getctime)
        return latest
    else:
        return dir_  # is actually a file
