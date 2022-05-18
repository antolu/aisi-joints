import time as t


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

