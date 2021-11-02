import types
import inspect
import functools
from typing import List, Callable

import numpy as np

from cellst.utils._types import Image, Mask, Track
from cellst.utils._types import INPT_NAMES


# Useful functions for interpolating nans in the data
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


# Interpolates along rows in 2D array
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


class ImageHelper():
    """
    Decorator to help with passing only the correct image stacks to functions
    """
    def __init__(self,
                 by_frame: bool = False,
                 dtype: type = None,
                 ) -> None:
        """
        TOOD:
            - For now by_frame will be boolean option. The issue with passing
              multiple frames to a function is that it will change the size
              of the stack. This isn't an issue, but I don't see it being
              useful right now.
            - add option to all overlapping the frames passed to functions
        """
        # if frame is not None, will be used to pass successive frames to func
        self.by_frame = by_frame
        self.dtype = dtype

    def __call__(self, func):
        # Get expected type from function annotation
        self.output_type = inspect.signature(func).return_annotation

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], (list, np.ndarray)):
                # This means the method is not static
                pass_to_func, nargs, nkwargs = self._type_helper(func,
                                                                 *args[1:],
                                                                 **kwargs)
                pass_to_func.insert(0, args[0])
            else:
                pass_to_func, nargs, nkwargs = self._type_helper(func,
                                                                 *args,
                                                                 **kwargs)

            # Function expects and returns a stack.
            if not self.by_frame or len(pass_to_func) == 0:
                stack = func(*pass_to_func, *nargs, **nkwargs)
            else:
                stack = self._pass_by_frames(func, pass_to_func,
                                             *nargs, **kwargs)
            # NOTE: won't work for 1D arrays. Does that matter?
            while stack.ndim < 3 and self.output_type in (Image, Mask, Track):
                stack = np.expand_dims(stack, axis=-1)

            return self.output_type, stack
        return wrapper

    def _type_helper(self, func, imgs, msks, trks, arrs, *args, **kwargs):
        '''
        This func is for sorting the input types and selecting the correct
        types that should get passed to the function.
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

    def _pass_by_frames(self,
                        func: Callable,
                        pass_to_func: List,
                        *args, **kwargs
                        ) -> List:

        # Use the output type to set the value of the array
        if self.dtype is not None:
            dtype = self.dtype
        elif self.output_type.__name__ == 'image':
            # If image, keep the input type
            dtype = pass_to_func[0].dtype
        elif self.output_type.__name__ == 'mask':
            # If mask, use only positive integers
            dtype = np.int16
        elif self.output_type.__name__ == 'track':
            dtype = np.uint16
        else:
            raise TypeError(f'Was unable to determine type for output of {func}.')

        # Initialize output array
        out = np.empty(pass_to_func[0].shape).astype(dtype)
        for fr in range(out.shape[0]):
            # Pass each frame individually for all inputs
            out[fr, ...] = func(*[p[fr, ...] for p in pass_to_func],
                                *args, **kwargs)

        return out
