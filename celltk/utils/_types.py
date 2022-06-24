import warnings
from typing import Tuple, TypeVar, Generic

import numpy as np


stack = TypeVar('stack')
class Stack(Generic[stack]):
    __args__ = None

    @classmethod
    def __class_getitem__(cls, vars):
        # Needs to make an instance so __args__
        # isn't overwritten for all uses of cls
        instance = super().__new__(cls)
        instance.__args__ = tuple([vars])
        instance.__name__ = cls.__name__
        return instance

    @classmethod
    def __init_subclass__(cls, *, name: str = None):
        super().__init_subclass__()
        if name: cls.__name__ = name


class Stack(Stack, name='stack'):
    pass


class Image(Stack, name='image'):
    pass


class Mask(Stack, name='mask'):
    pass


class Array(Stack, name='array'):
    pass

class Optional(Stack, name='optional'):
    pass


# Save input names and types
INPT_NAMES = [Image.__name__, Mask.__name__,
              Array.__name__, Stack.__name__]
_INPT_NAMES_NO_STACK = [Image.__name__, Mask.__name__,
                        Array.__name__]
INPT_NAME_IDX = {n: i for i, n in enumerate(INPT_NAMES)}
INPT = [Image, Mask, Array]
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
        if key[1] == 'stack':
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


class RandomNameProperty:
    """
    This class is to be used with skimage.regionprops_table.
    Every extra property passed to regionprops_table must
    have a unique name, however, I want to use several as a
    placeholder, so that I can get the right size array, but fill
    in the values later. So, this assigns a random __name__.

    NOTE:
        - This does not guarantee a unique name, so getting IndexError
          in Extract is still possible.
    """
    def __init__(self) -> None:
        # Use ranndom int as name of the property
        rng = np.random.default_rng()
        self.__name__ = str(rng.integers(999999))

    @staticmethod
    def __call__(empty):
        return np.nan
