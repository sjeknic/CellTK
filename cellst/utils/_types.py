from typing import NewType

import numpy as np

from cellst.core.arrays import Condition


class ImageContainer(dict):
    """
    Class to hold all image stacks.

    For now just a dictionary, but class is different
    so that ImageHelper knows what to do.

    TODO:
        - Add check on set_item to make sure it is ndarray
    """


# Define custom types to make output tracking esier
Image = NewType('image', np.ndarray)
Mask = NewType('mask', np.ndarray)
Track = NewType('track', np.ndarray)
Arr = NewType('array', Condition)  # For Condition/Experiment
Same = NewType('same', np.ndarray)  # Allows functions to return multiple types

# Save input names and types
INPT_NAMES = [Image.__name__, Mask.__name__, Track.__name__, Arr.__name__]
INPT_NAME_IDX = {n: i for i, n in enumerate(INPT_NAMES)}
INPT = [Image, Mask, Track, Arr]
INPT_IDX = {n: i for i, n in enumerate(INPT)}
TYPE_LOOKUP = dict(zip(INPT_NAMES, INPT))
