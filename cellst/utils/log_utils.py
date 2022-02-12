import os
import logging

import numpy as np


def get_logger(name: str,
               path: str = None,
               filename: str = 'log.txt',
               file_level: str = 'info',
               console_level: str = 'warning',
               overwrite: bool = True
               ) -> logging.Logger:
    """
    Used to get root-level loggers
    """
    # Get logger with name of module
    logger = logging.getLogger(name)

    # Confirm logger is unique
    while logger.handlers:
        app = np.random.randint(10000)
        logger = logging.getLogger(f'{app}>{name}')

    # Set parameters
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s: %(message)s')

    # Create file handler to write outputs
    path = '.' if path is None else path
    mode = 'w' if overwrite else 'a'
    fh = logging.FileHandler(os.path.join(path, filename), mode=mode)
    fh.setFormatter(formatter)
    fh.addFilter(CustomNameFilter())
    try:
        fh.setLevel(getattr(logging, file_level.upper()))
    except AttributeError:
        raise ValueError(f'Not a valid logging level: {file_level}')

    # Create stream handler to specify standard output
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(CustomNameFilter())
    sh.addFilter(SkimageWarningFilter())
    try:
        sh.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        raise ValueError(f'Not a valid logging level: {console_level}')

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Set up warnings logger
    warning_logger = get_warning_logger()
    warning_logger.addHandler(fh)
    warning_logger.addHandler(sh)

    return logger


def get_warning_logger() -> logging.Logger:
    """
    Set up a logger to capture warnings and log them
    """
    logging.captureWarnings(True)
    warning_logger = logging.getLogger('py.warnings')
    if warning_logger.handlers: warning_logger.handlers = []

    return warning_logger


def get_console_logger(level: str = 'WARNING') -> logging.Logger:
    """
    Used as a default logger. Writes to console only.
    """
    # Get logger with name of module
    logger = logging.getLogger('console')

    # Set parameters
    logger.setLevel(getattr(logging, level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s: %(message)s')

    # Create stream handler to specify standard output
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, level))
    sh.setFormatter(formatter)

    # Add handler and return
    logger.addHandler(sh)
    return logger


def get_null_logger() -> logging.Logger:
    """
    Does not log any outputs
    """
    # Get logger with name of module
    logger = logging.getLogger('null')
    handler = logging.NullHandler()
    logger.addHandler(handler)

    return logger


class CustomNameFilter(logging.Filter):
    """
    Used to remove the custom marker added to filters
    """
    def filter(self, record):
        record.name = record.name.split('>')[-1]
        return True


class SkimageWarningFilter(logging.Filter):
    """
    Used to silence warnings raised by downcasting
    in skimage.util functions
    """
    def __init__(self) -> None:
        self.seen = 0

    def filter(self, record):
        if 'Downcasting' in record.message:
            if self.seen >= 1:
                return False
            self.seen += 1

            return True
        else:
            return True
