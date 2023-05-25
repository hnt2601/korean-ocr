import os
import logging
from logging.handlers import TimedRotatingFileHandler

from src.config import NAME

log_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(log_dir, f"../data/{NAME}")
os.makedirs(log_dir, exist_ok=True)
formatter = logging.Formatter(
    "%(asctime)s %(filename)s:%(lineno)d\t %(levelname)s: %(message)s"
)


def setup_logger(name, debug=False):
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_file = os.path.join(log_dir, name)
    tfh = TimedRotatingFileHandler(log_file, when="midnight", interval=1)
    tfh.setFormatter(formatter)
    tfh.suffix = "%Y-%m-%d"
    logger.addHandler(tfh)

    return logger
