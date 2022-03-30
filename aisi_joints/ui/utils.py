"""
A powerful, synchronous implementation of run_in_main_thread(...).
It allows you to receive results from the function invocation:

    @run_in_main_thread
    def return_2():
        return 2

    # Runs the above function in the main thread and prints '2':
    print(return_2())
"""

import typing as t
from functools import wraps
from threading import Event, get_ident, Thread

from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import QApplication, QFileDialog, QListView, QTreeView, QFileSystemModel, QAbstractItemView, \
    QDialog


def thread(function: t.Callable, *args, **kwargs):
    th = Thread(target=function, args=args, kwargs=kwargs)
    th.start()

    return th


def run_in_thread(thread_fn: QThread):
    def decorator(f):
        @wraps(f)
        def result(*args, **kwargs):
            thread = thread_fn()
            return Executor.instance().run_in_thread(thread, f, args, kwargs)

        return result

    return decorator


def _main_thread():
    app = QApplication.instance()
    if app:
        return app.thread()
    # We reach here in tests that don't (want to) create a QApplication.
    if int(QThread.currentThreadId()) == get_ident():
        return QThread.currentThread()
    raise RuntimeError('Could not determine main thread')


run_in_main_thread = run_in_thread(_main_thread)


def is_in_main_thread():
    return QThread.currentThread() == _main_thread()


class Executor:
    _INSTANCE = None

    @classmethod
    def instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(QApplication.instance())
        return cls._INSTANCE

    def __init__(self, app: QApplication):
        self._pending_tasks = []
        self._app_is_about_to_quit = False
        app.aboutToQuit.connect(self._about_to_quit)

    def _about_to_quit(self):

        self._app_is_about_to_quit = True
        for task in self._pending_tasks:
            task.set_exception(SystemExit())
            task.has_run.set()

    def run_in_thread(self, qthread: QThread, f: t.Callable, args: t.Tuple,
                      kwargs: t.Dict):
        if QThread.currentThread() == qthread:
            return f(*args, **kwargs)
        elif self._app_is_about_to_quit:
            # In this case, the target thread's event loop most likely is not
            # running any more. This would mean that our task (which is
            # submitted to the event loop via events/slots) is never run.
            raise SystemExit()
        task = Task(f, args, kwargs)
        self._pending_tasks.append(task)
        try:
            receiver = Receiver(task)
            receiver.moveToThread(qthread)
            sender = Sender()
            sender.signal.connect(receiver.slot)
            sender.signal.emit()
            if not qthread.isRunning():
                qthread.start()
            task.has_run.wait()
            return task.result
        finally:
            self._pending_tasks.remove(task)


class Task:
    def __init__(self, fn, args, kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.has_run = Event()
        self._result = self._exception = None

    def __call__(self):
        try:
            self._result = self._fn(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e
        finally:
            self.has_run.set()

    def set_exception(self, exception):
        self._exception = exception

    @property
    def result(self):
        if not self.has_run.is_set():
            raise ValueError("Hasn't run.")
        if self._exception:
            raise self._exception
        return self._result


def choose_directories(parent: QObject) -> t.List[str]:
    dialog = QFileDialog(parent)
    dialog.setWindowTitle('Select directories with images')
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.DirectoryOnly)
    for view in dialog.findChildren(
            (QListView, QTreeView)):
        if isinstance(view.model(), QFileSystemModel):
            view.setSelectionMode(
                QAbstractItemView.ExtendedSelection)
    dialog.deleteLater()

    if dialog.exec() == QDialog.Accepted:
        return dialog.selectedFiles()
    else:
        return []
