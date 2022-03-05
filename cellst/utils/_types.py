import warnings
from typing import NewType, Tuple

import numpy as np

from cellst.core.arrays import ConditionArray


# Define custom types to make output tracking esier
Image = NewType('image', np.ndarray)
Mask = NewType('mask', np.ndarray)
Track = NewType('track', np.ndarray)
Arr = NewType('array', ConditionArray)  # For ConditionArray/ExperimentArray
Same = NewType('same', np.ndarray)  # Allows functions to return multiple types


# Save input names and types
INPT_NAMES = [Image.__name__, Mask.__name__, Track.__name__, Arr.__name__, Same.__name__]
_INPT_NAMES_NO_STACK = [Image.__name__, Mask.__name__, Track.__name__, Arr.__name__]
INPT_NAME_IDX = {n: i for i, n in enumerate(INPT_NAMES)}
INPT = [Image, Mask, Track, Arr]
INPT_IDX = {n: i for i, n in enumerate(INPT)}
TYPE_LOOKUP = dict(zip(INPT_NAMES, INPT))

# Split key should be saved in a global location
_split_key = '&'


class ImageContainer(dict):
    """
    Class to hold all image stacks.

    For now just a dictionary, but class is different
    so that ImageHelper knows what to do.

    TODO:
        - Add check on set_item to make sure it is ndarray
    """
    def __getitem__(self, key: Tuple[str]):
        # Check the expected type
        if key[1] == 'same':
            out = []
            for t in _INPT_NAMES_NO_STACK:
                # If the type is Stack, find all inputs with matching key
                try:
                    out.append(super().__getitem__((key[0], t)))
                except KeyError:
                    pass

            if not out:
                raise KeyError(key)
            elif len(out) == 1:
                return out[0]
            else:
                # Found more than one
                warnings.warn(f'Found multiple stacks for {key}: {out} '
                              'Returning only the first.')
                return out[0]
        else:
            return super().__getitem__(key)


class RandomNameProperty():
    """
    This class is to be used with skimage.regionprops_table.
    Every extra property passed to regionprops_table must
    have a unique name, however, I want to use several as a
    placeholder, so that I can get the right size array, but fill
    in the values later. So, this assigns a random __name__.

    NOTE:
        - This does not guarantee a unique name, so getting IndexError
          in Extractor is still possible.
    """
    def __init__(self) -> None:
        # Use ranndom int as name of the property
        rng = np.random.default_rng()
        self.__name__ = str(rng.integers(999999))

    @staticmethod
    def __call__(empty):
        return np.nan



