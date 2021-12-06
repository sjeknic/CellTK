import os
import logging

def get_logger(name: str,
               path: str = None,
               filename: str = 'log.txt',
               overwrite: bool = True) -> logging.Logger:
    """
    """
    # Get logger with name of module
    logger = logging.getLogger(name)

    # Set parameters
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    # Create file handler to write outputs
    path = '.' if path is None else path
    mode = 'w' if overwrite else 'a'
    fh = logging.FileHandler(os.path.join(path, filename), mode=mode)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Create stream handler to specify standard output
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Initialize log file and return
    logger.info('Log initiated')
    return logger


def get_console_logger(level: str = 'WARNING') -> logging.Logger:
    """
    Used as a default logger. Writes to console only.
    """
    # Get logger with name of module
    logger = logging.getLogger('console')

    # Set parameters
    logger.setLevel(getattr(logging, level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    # Create stream handler to specify standard output
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # Add handler and return
    logger.addHandler(sh)
    return logger
