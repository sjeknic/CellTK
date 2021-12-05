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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler to write outputs
    path = '.' if path is None else path
    mode = 'w' if overwrite else 'a'
    fh = logging.FileHandler(os.path.join(path, filename), mode=mode)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)

    # Initialize log file and return
    logger.info('Log initiated')
    return logger


def get_empty_logger() -> logging.Logger:
    """
    TODO: Should probably just add more options to the above
    """
    # Get logger with name of module
    logger = logging.getLogger('empty')
    logger.setLevel(logging.ERROR)

    return logger