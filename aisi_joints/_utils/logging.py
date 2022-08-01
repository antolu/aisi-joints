import logging

__all__ = ["setup_logger"]


def setup_logger(debug: bool = False, file_logger: str = None):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()

    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if file_logger is not None:
        fh = logging.FileHandler(file_logger)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        log.addHandler(fh)
