import sys
import os
import types
import inspect
import functools
import contextlib
import warnings
from typing import List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cellst.utils._types import Image, Mask, Track, Arr
from cellst.utils._types import INPT_NAMES
from cellst.utils.operation_utils import sliding_window_generator


def nan_helper(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of nans in a 1D array."""
    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_helper_2d(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of nans along rows in 2D array."""
    temp = np.zeros(arr.shape)
    temp[:] = np.nan
    for n, y in enumerate(arr.copy()):
        nans, z = nan_helper(y)
        y[nans] = np.interp(z(nans), z(~nans), y[~nans])
        temp[n, :] = y


def folder_name(path: str) -> str:
    """Returns name of last folder in a path"""
    return os.path.basename(os.path.normpath(path))


# Decorator that end users can use to add custom functions
# to Operations.
# TODO: Needs to wrap in ImageHelper
def custom_function(operation):
    def decorator(func):
        func = types.MethodType(func, operation)

        if not hasattr(operation, func.__name__):
            setattr(operation, func.__name__, func)
        else:
            raise ValueError(f'Function {func} already exists in {operation}.')

        return func
    return decorator


# Functions to block output to Terminal
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    Context manager to redirect outputs from non-CellST funcs to os.devnull
    """
    if stdout is None: stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


class ImageHelper():
    """
    Decorator to help with passing only the correct image stacks to functions

    TODO:
        - See PEP563, should use typing.get_type_hints
    """
    def __init__(self,
                 by_frame: bool = False,
                 overlap: int = 0,
                 dtype: type = None,
                 copy_to_output: type = None
                 ) -> None:
        """
        overlap: int(amount of frames to overlap between passing)
            e.g. overlap = 1: [0, 1], [1, 2], [2, 3], [3, 4]
                 overlap = 2: [0, 1, 2], [1, 2, 3], [2, 3, 4]

        NOTE: Overlaps get passed as a stack, not as separate args.
              i.e. if overlap = 1, image.shape = (2, h, w)
        """
        self.by_frame = by_frame
        self.overlap = overlap
        self.dtype = dtype

        # Get copy_to_output as str
        if isinstance(copy_to_output, str) or copy_to_output is None:
            self.copy_to_output = copy_to_output
        else:
            try:
                self.copy_to_output = copy_to_output.__name__
            except AttributeError:
                raise ValueError('Did not understand copy_to_output type.')

    def __call__(self, func):
        # Get expected type from function annotation
        self.output_type = inspect.signature(func).return_annotation
        self.func = func

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            # Find all of the non-image related inputs (e.g. class, self, etc.)
            # Save the nonarr_inputs and pass others to _type_helper
            nonarr_idx = 0
            while True:
                if not isinstance(args[nonarr_idx], (list, np.ndarray)):
                    nonarr_idx += 1
                else:
                    break
            nonarr_inputs = args[:nonarr_idx]
            args = args[nonarr_idx:]

            # Sort the inputs and keep only those that are relevant
            pass_to_func, nargs, nkwargs = self._type_helper(*args, **kwargs)

            # Function expects and returns a stack.
            if not self.by_frame or len(pass_to_func) == 0:
                stack = self.func(*nonarr_inputs, *pass_to_func,
                                  *nargs, **nkwargs)
            else:
                stack = self._pass_by_frames(pass_to_func, nonarr_inputs,
                                             *nargs, **kwargs)

            # NOTE: won't work for 1D arrays. Does that matter?
            while stack.ndim < 3 and self.output_type in (Image, Mask, Track):
                stack = np.expand_dims(stack, axis=-1)

            return self.output_type, stack
        return wrapper

    def _type_helper(self, imgs, msks, trks, arrs, *args, **kwargs):
        """
        This func is for sorting the input types and selecting the correct
        types that should get passed to the function.
        """
        # Check which inputs the function is expecting and only pass those
        expected_types = [i.annotation.__name__
                          for i in inspect.signature(self.func).parameters.values()
                          if hasattr(i.annotation, '__name__')]
        expected_names = [i.name
                          for i in inspect.signature(self.func).parameters.values()]

        # Check for what the function expects. Include plurals for the name
        inpt_bools = [(i in expected_types)
                      or (i in expected_names) or (i + 's' in expected_names)
                      for i in INPT_NAMES]
        pass_to_func = [i for b, inpt in zip(inpt_bools, [imgs, msks, trks, arrs])
                        for i in inpt if b]

        # Save input types for future use
        self.input_type_idx = [i for b, i, count
                               in zip(inpt_bools,
                                      INPT_NAMES,
                                      [imgs, msks, trks, arrs])
                               for c in count if b]


        return pass_to_func, args, kwargs

    def _guess_input_from_output(self, output_type: type) -> type:
        """
        Uses simply heuristics to guess the input type
        """
        if output_type == 'image':
            # For image, assume image
            return 'image'
        elif output_type == 'mask' or output_type == 'track':
            # For track or mask, I think mask is best
            return 'mask'
        elif output_type == 'array':
            # Not sure this would ever happen
            return 'array'

    def _get_output_array(self, pass_to_func: List[np.ndarray]) -> np.ndarray:
        """
        """
        # Use the output type to set the value of the array
        if self.dtype is not None:
            dtype = self.dtype
        elif self.output_type.__name__ == 'image':
            # If image, keep the input type
            dtype = pass_to_func[0].dtype
        elif self.output_type.__name__ == 'mask':
            # If mask, use only positive integers
            dtype = np.uint16
        elif self.output_type.__name__ == 'track':
            dtype = np.int16
        else:
            raise TypeError('Was unable to determine type for '
                            f'output of {self.func}.')

        # Make output array
        # NOTE: This assumes that output is the same shape as input
        #       and that all the inputs have the same shape
        out = np.empty(pass_to_func[0].shape).astype(dtype)

        '''If overlap > 0, frames need to be copied to the output array.
        If user hasn't specified which input to copy, then guess based
        on output.'''
        if self.overlap > 0:
            if self.copy_to_output is None:
                # If only one input, then obviously it has to be used
                if len(pass_to_func) == 1:
                    copy_idx = 0
                else:
                    warnings.warn('If overlap is greater than 0, specify '
                                  'copy_to_output. Trying to guess based on '
                                  'output type.', UserWarning)
                    copy_type = self._guess_output_from_input()
                    try:
                        copy_idx = self.input_type_idx.index(copy_type)
                    except ValueError:
                        raise ValueError(f'Did not find type {copy_type} '
                                         'to add to output array. '
                                         'Set self.copy_to_output.')
            else:
                copy_idx = self.input_type_idx.index(self.copy_to_output)

            # Copy number of frames from the correct input
            frames = pass_to_func[copy_idx][:self.overlap]
            out[:self.overlap, ...] = frames

        return out

    def _pass_by_frames(self,
                        pass_to_func: List,
                        nonarr_inputs: Tuple,
                        *args, **kwargs
                        ) -> np.ndarray:
        # Initialize output array - with frames copied if needed
        out = self._get_output_array(pass_to_func)

        # Get shape of window, slides along frame axis (axis 0)
        window_shape = (self.overlap + 1, *pass_to_func[0].shape[1:])
        windows = [sliding_window_generator(p, window_shape)
                   for p in pass_to_func]

        for fr, win in enumerate(zip(*windows)):
            # Pass each generator and save in index + overlap
            idx = fr + self.overlap
            out[idx, ...] = self.func(*nonarr_inputs,
                                      *win, *args, **kwargs)

        return out
