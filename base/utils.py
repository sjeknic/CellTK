import types
import inspect
from typing import NewType

import numpy as np

from base.custom_array import CustomArray

# Define custom types to make output tracking esier
Image = NewType('image', np.ndarray)
Mask = NewType('mask', np.ndarray)
Track = NewType('track', np.ndarray)

INPT_NAMES = [Image.__name__, Mask.__name__, Track.__name__, CustomArray.__name__]
INPT_NAME_IDX = {n: i for i, n in enumerate(INPT_NAMES)}
INPT = [Image, Mask, Track, CustomArray]
INPT_IDX = {n: i for i, n in enumerate(INPT)}

# Useful functions for interpolating nans in the data
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_helper_2d(arr):
    #probably can be done faster
    temp = np.zeros(arr.shape)
    temp[:] = np.nan
    for n, y in enumerate(arr.copy()):
        nans, z = nan_helper(y)
        y[nans] = np.interp(z(nans), z(~nans), y[~nans])
        temp[n, :] = y


# Decorator that end users can use to add custom functions
# to Operations.
def custom_function(operation):
    def decorator(func):
        func = types.MethodType(func, operation)

        if not hasattr(operation, func.__name__):
            setattr(operation, func.__name__, func)
        else:
            raise ValueError(f'Function {func} already exists in {operation}.')

        return func
    return decorator


# Decorator to help with passing only the correct image stacks to functions
def image_helper(func):
    """
    Passes only the required image stacks to the function
    Returns the expected output type as well as the result

    TODO:
        - Make a version or add to this, a way to pass successive slices
          or a generator that passes successive slices
        - I think this might need to be a class and or I need a __repr__ in order to
          still see which operation_function is being called. Otherwise, print
          will just show this decorator for all functions
    """
    # Determine the expected output type
    output_type = inspect.signature(func).return_annotation

    def decorator(*args, **kwargs):
        '''This function is so that this decorator can work
        for static and nonstatic methods. This function gets
        the correct images from _type_helper and then passes to the
        function'''

        # imgs passed to this function should always be list, use that to
        # check for non-static methods.
        if not isinstance(args[0], (list, np.ndarray)):
            pass_to_func, nargs, nkwargs = _type_helper(*args[1:], **kwargs)
            pass_to_func.insert(0, args[0])
        else:
            pass_to_func, nargs, nkwargs = _type_helper(*args, **kwargs)

        # Because this function is expecting a stack, it should always return a stack
        stack = func(*pass_to_func, *nargs, **nkwargs)
        # NOTE: won't work for 1D images. Does that matter?
        while stack.ndim < 3 and output_type in (Image, Mask, Track):
            stack = np.expand_dims(stack, axis=-1)

        return output_type, stack

    def _type_helper(imgs, msks, trks, arrs, *args, **kwargs):
        '''
        This func is now just for sorting the input types
        '''
        # Check which inputs the function is expecting and only pass those
        expected_types = [i.annotation.__name__
                          for i in inspect.signature(func).parameters.values()
                          if hasattr(i, __name__)]
        expected_names = [p.name
                          for p in inspect.signature(func).parameters.values()]

        inpt_bools = [(i in expected_types) or (i in expected_names)
                      for i in INPT_NAMES]
        pass_to_func = [i for b, inpt in zip(inpt_bools, [imgs, msks, trks, arrs])
                        for i in inpt if b]
        return pass_to_func, args, kwargs

    return decorator
