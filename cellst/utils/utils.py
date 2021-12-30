import sys
import os
import types
import inspect
import functools
import contextlib
import warnings
import logging
from typing import List, Tuple

import numpy as np

from cellst.utils._types import (Image, Mask, Track,
                                 Arr, ImageContainer,
                                 Condition, Experiment,
                                 INPT_NAMES)
from cellst.utils.operation_utils import sliding_window_generator
from cellst.utils.log_utils import get_null_logger


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
        - Add logger
    """
    __name__ = 'ImageHelper'

    def __init__(self,
                 by_frame: bool = False,
                 overlap: int = 0,
                 as_tuple: bool = False,
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
        # Save inputs
        self.by_frame = by_frame
        self.overlap = overlap
        self.dtype = dtype
        self.as_tuple = as_tuple

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
        self.func = func
        self.output_type = inspect.signature(self.func).return_annotation

        func_params = inspect.signature(self.func).parameters.values()
        self.expected_types = [i.annotation.__name__
                               for i in func_params
                               if hasattr(i.annotation, '__name__')]
        self.expected_names = [i.name for i in func_params]

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            # Sort args and kwargs into the respective values
            if isinstance(args[0], ImageContainer):
                # If first value is not self or class, then staticmethod
                img_container = args[0]
                calling_cls = []
                args = args[1:]
            else:
                calling_cls = [args[0]]
                img_container = args[1]
                args = args[2:]

            self.logger = self._get_calling_logger(calling_cls)

            # Sort the inputs and keep only those that are relevant
            keys, pass_to_func, nkwargs = self._type_helper(img_container,
                                                            **kwargs)

            # Function expects and returns a stack.
            if not self.by_frame or len(pass_to_func) == 0:
                # Pass all inputs together
                stack = self.func(*calling_cls, *pass_to_func,
                                  *args, **nkwargs)
            else:
                # Pass inputs individually
                stack = self._pass_by_frames(pass_to_func, calling_cls,
                                             *args, **nkwargs)

            # Get the correct outputs and keys before returning
            keys, stack = self._correct_outputs(keys, stack, calling_cls)

            return keys, stack
        return wrapper

    def _type_helper(self, img_container, **kwargs):
        """
        This func is for sorting the input types and selecting the correct
        types that should get passed to the function.
        """
        # First check if user specified names
        img_container, kwargs = self._name_helper(img_container, **kwargs)

        # TODO: Should names continue to be allowed to be used as the input?
        # TODO: I'm not sure inpt_bools is even necessary after _name_helper
        # Check for what the function expects. Include plurals for the name
        inpt_bools = [(i in self.expected_types)
                      or (i in self.expected_names)
                      or (i + 's' in self.expected_names)
                      for i in INPT_NAMES]

        self.logger.info('Function accepts: '
                         f'{[i for i, b in zip(INPT_NAMES, inpt_bools) if b]}')

        imgs, msks, trks, arrs = ([(k, v) for k, v in img_container.items()
                                   if k[1] == i]
                                  for i in INPT_NAMES)

        # Check that everything here works
        if self.as_tuple:
            # If input doesn't exist, nothing is appended
            keys, pass_to_func = zip(*[zip(*inpt) for b, inpt
                                       in zip(inpt_bools,
                                              [imgs, msks, trks, arrs])
                                       if b and len(inpt) > 0])
            inpt_size_type = [(p.shape, p.dtype) for tup in pass_to_func
                              for p in tup]
        else:
            keys, pass_to_func = zip(*[i for b, inpt
                                       in zip(inpt_bools,
                                              [imgs, msks, trks, arrs])
                                       for i in inpt if b])
            inpt_size_type = [(p.shape, p.dtype) for p in pass_to_func]

        self.input_type_idx = [i for b, i, count
                               in zip(inpt_bools,
                                      INPT_NAMES,
                                      [imgs, msks, trks, arrs])
                               for c in count if b]

        # Flatten keys, as outputs must be flat
        keys = tuple([k for sk in keys for k in sk])

        self.logger.info(f'Selected inputs: {list(zip(keys, inpt_size_type))}')

        return keys, pass_to_func, kwargs

    def _name_helper(self, img_container, **kwargs):
        """
        Finds images the user specified using kwargs
        """
        new_container = ImageContainer()

        for exp_name, exp_typ in zip(self.expected_names, self.expected_types):
            # First check if user provided the inputs
            if (exp_name in kwargs) and (exp_typ in INPT_NAMES):
                # Then check if they provided one name or multiple
                names = kwargs[exp_name]
                if isinstance(names, str):
                    # Append the expected type to the name
                    names = [(names, exp_typ)]
                else:
                    names = [(nm, exp_typ) for nm in names]

                # Remove from kwargs
                del kwargs[exp_name]

            else:
                # If user did not provide name, load all images of exp_typ
                names = [k for k in img_container if k[1] == exp_typ]

            # Load image stack for each name
            for nm in names:
                try:
                    new_container[nm] = img_container[nm]
                except KeyError:
                    # TODO: Add a strict_type option. If False, check for nm
                    #       of different types.
                    raise KeyError(f'Could not find input {nm[0]} of '
                                   f'type {nm[1]} in the inputs to '
                                   f'function {self.func}')

        return new_container, kwargs

    def _correct_outputs(self, keys, stack, calling_cls=[]):
        # Store keys as list if not already
        if isinstance(keys[0], str):
            # Output type defined by function
            keys = (keys[0], self.output_type.__name__)
            keys = [keys]

        # Store stack as list if not already
        if isinstance(stack, np.ndarray):
            stack = [stack]
        elif isinstance(stack, (Condition, Experiment)):
            stack = [stack]
            # Assume only one output key - set to output id
            try:
                keys = [calling_cls[0].output_id]
            except IndexError:
                # The method was a staticmethod use first key and warn user
                keys = [keys[0]]
                warnings.warn('Possible mismatch with keys for ',
                              f'for {self.func.__name__}.')

        # Check that length matches
        if len(stack) != len(keys):
            # TODO: Is there a use-case for this or is error fine?
            raise ValueError(f'Length of outputs ({len(stack)}) does not '
                             f'match length of keys ({len(keys)}).')

        # Adjust array dimensions if needed
        for n, (k, st) in enumerate(zip(keys, stack)):
            while st.ndim < 3 and any([i in k for i in (Image, Mask, Track)]):
                st = np.expand_dims(st, axis=-1)
            stack[n] = st

        return keys, stack

    def _get_calling_logger(self, calling_cls):
        """
        Gets the operation logger to record info about inputs and outputs

        TODO:
            - I think this could be cleaner
        """
        if len(calling_cls) > 0:
            try:
                logger = calling_cls[0].logger
            except AttributeError:
                return get_null_logger()

            if logger is not None:
                return logging.getLogger(f'{logger.name}.{self.__name__}')
            else:
                return get_null_logger()

    def _guess_input_from_output(self, output_type: type) -> type:
        """
        Uses simple heuristics to guess the input type
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

    def _get_output_array(self, ex_arr: np.ndarray) -> np.ndarray:
        """
        TODO: Update to expect an array as input, not a list.
        TODO: copy_to_output is probably dumb and I'll ignore it for now.
        TODO: I don't think self.input_type_idx still works with inputs
        """
        # Use the output type to set the value of the array
        if self.dtype is not None:
            dtype = self.dtype
        elif self.output_type.__name__ == 'image':
            # If image, keep the input type
            dtype = ex_arr.dtype
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
        out = np.empty(ex_arr.shape).astype(dtype)

        '''If overlap > 0, frames need to be copied to the output array.
        If user hasn't specified which input to copy, then guess based
        on output.'''
        # TODO: copy_to_output needs to be handled differently
        if self.overlap > 0:
            if self.copy_to_output is None:
                # If only one input, then obviously it has to be used
                if len(ex_arr) == 1:
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
            frames = ex_arr[copy_idx][:self.overlap]
            out[:self.overlap, ...] = frames

        return out

    def _pass_by_frames(self,
                        pass_to_func: List,
                        calling_cls: Tuple,
                        *args, **kwargs
                        ) -> np.ndarray:
        """
        TODO:
            - How to handle multiple outputs? Probably best to do a test-run,
              then make the output arrays, then finish the run
        """
        # Initialize output array - with frames copied if needed
        if self.as_tuple:
            out = self._get_output_array(pass_to_func[0][0])
        else:
            out = self._get_output_array(pass_to_func[0])

        # Get generators for each array in pass_to_func
        windows = [sliding_window_generator(p, self.overlap)
                   for p in pass_to_func]

        for fr, win in enumerate(zip(*windows)):
            # Pass each generator and save in index + overlap
            idx = fr + self.overlap
            out[idx, ...] = self.func(*calling_cls,
                                      *win, *args, **kwargs)

        return out
