import logging


def initial_logger():
    FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger
