import logging


def setup_logger(debug: bool = False):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()

    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    log.addHandler(ch)
