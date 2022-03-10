import logging


def setup_logger():
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    log.addHandler(ch)
