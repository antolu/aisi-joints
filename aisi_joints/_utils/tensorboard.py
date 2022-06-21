import logging
import os
import os.path as path

from tensorboard import program, default

__all__ = ['TensorBoardTool']


class TensorBoardTool:
    def __init__(self, log_dir: str):
        self.log_dir: str = log_dir

    def run(self):
        if not path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Suppress http messages
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins())
        tb.configure(argv=[None, '--logdir', self.log_dir, '--bind_all'])

        url = tb.launch()
        print(f'TensorBoard at {url}')
