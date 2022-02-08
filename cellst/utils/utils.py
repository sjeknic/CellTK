import sys
import os
import inspect
import functools
import itertools
import contextlib
import warnings
import logging
from typing import List, Tuple, Generator

import numpy as np

from cellst.utils._types import (Image, Mask, Track,
                                 Arr, ImageContainer,
                                 INPT_NAMES)
from cellst.core.arrays import ConditionArray, ExperimentArray
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
    __name__ = 'ImageHelper'

    def __init__(self, *,
                 by_frame: bool = False,
                 as_tuple: bool = False,
                 dtype: type = None,
                 ) -> None:
        """
        """
        # Save inputs
        self.by_frame = by_frame
        self.dtype = dtype
        self.as_tuple = as_tuple

    def __call__(self, func):
        # Save information about the function from the signature
        self.func = func
        self.output_type = inspect.signature(self.func).return_annotation

        func_params = inspect.signature(self.func).parameters.values()
        self.expected_types = [i.annotation.__name__
                               for i in func_params
                               if hasattr(i.annotation, '__name__')]
        self.expected_names = [i.name for i in func_params]

        # Used to determine how many of each input to pass
        self.expected_numbers = {i: len([l for l in self.expected_types if l == i])
                                 for i in INPT_NAMES}

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
            self.logger.info(f'ImageHelper called for {self.func.__name__}')
            self.logger.info(f'by_frame: {self.by_frame}, '
                             f'as_tuple: {self.as_tuple}, dtype: {self.dtype}')

            # Sort the inputs and keep only those that are relevant
            keys, pass_to_func, nkwargs = self._type_helper(img_container,
                                                            **kwargs)

            # Function expects and returns a stack.
            if not self.by_frame or not pass_to_func:
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

        # Check for what the function expects. Include plurals for the name
        inpt_bools = [(i in self.expected_types) for i in INPT_NAMES]
        self.logger.info('Function accepts: '
                         f'{[i for i, b in zip(INPT_NAMES, inpt_bools) if b]}')

        # Get the inputs from the selected img_container
        imgs, msks, trks, arrs = ([(k, v) for k, v in img_container.items()
                                   if k[1] == i]
                                  for i in INPT_NAMES)

        # Sort the inputs based on what is requested
        keys = []
        pass_to_func = []
        inpt_size_type = []
        comb_inputs = zip([imgs, msks, trks, arrs],
                          inpt_bools,
                          INPT_NAMES)
        for inpt, incl, typ in comb_inputs:
            if incl:
                # Check how inputs are to be passed
                if self.as_tuple:
                    # Inputs are not trimmed or sorted if passed as tuple
                    pass_to_func.append(tuple([i[1] for i in inpt]))
                else:
                    # Trim the input based on the number to be passed
                    # Reverse order so most recent entry is passed first
                    inpt = inpt[-self.expected_numbers[typ]:][::-1]
                    pass_to_func.extend([i[1] for i in inpt])

                # These are just used for naming and logging, so always flat
                keys.extend([i[0] for i in inpt])
                inpt_size_type.extend([(i[1].shape, i[1].dtype) for i in inpt])

        # TODO: Remove this and fix copy_to_output
        self.input_type_idx = [i for b, i, count
                               in zip(inpt_bools,
                                      INPT_NAMES,
                                      [imgs, msks, trks, arrs])
                               for c in count if b]

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
                if names is None:
                    # User does not want any image passed
                    names = []
                elif isinstance(names, str):
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
                # The names in this container use the INPUT type
                try:
                    new_container[nm] = img_container[nm]
                except KeyError:
                    # TODO: Add a strict_type option. If False, check for nm
                    #       of different types.
                    self.logger.info(f'Available images of type {nm[1]}: '
                                     f'{[i for i in img_container if i[1] == nm[1]]}')
                    raise KeyError(f'Could not find input {nm[0]} of '
                                   f'type {nm[1]} in the inputs to '
                                   f'function {self.func}')

        return new_container, kwargs

    def _correct_outputs(self, keys, stack, calling_cls=[]):
        # Store keys as list if not already
        if isinstance(keys[0], str):
            keys = [keys]

        # If output_type is same out_type = in_type, else defined by function
        if self.output_type.__name__ != 'same':
            keys = [(k[0], self.output_type.__name__) for k in keys]

        # Store stack as list if not already
        if isinstance(stack, np.ndarray):
            stack = [stack]
        elif isinstance(stack, tuple):
            stack = list(stack)
        elif isinstance(stack, (ConditionArray, ExperimentArray)):
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
        if len(stack) < len(keys):
            # TODO: This doesn't work with Same type output
            # If fewer images than keys, try to match by type
            same_type_keys = [k for k in keys
                              if k[1] == self.output_type.__name__]
            same_type_keys = same_type_keys[:len(stack)]

            if len(same_type_keys) < len(stack):
                # Not enough of same type
                keys = keys[:len(stack)]
            else:
                keys = same_type_keys

        elif len(stack) > len(keys):
            # Returned more images than inpupts... Will deal with if it happens
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

    def _pass_by_frames(self,
                        pass_to_func: List,
                        calling_cls: Tuple,
                        *args, **kwargs
                        ) -> np.ndarray:
        """
        Creates output array and passes individual frames to func

        TODO:
            - Is it okay to convert float64 -> float32? or int32 -> int16
            - So this actually won't work if as_tuple and by_frame
              are both true, because the sliding window generator
              isn't set up to accept tuples.
        """
        # Get window generators for each array in pass_to_func
        if self.as_tuple:
            out_shape = pass_to_func[0][0].shape
            windows = [self._multiple_sliding_windows(p) for p in pass_to_func]
        else:
            out_shape = pass_to_func[0].shape
            windows = [sliding_window_generator(p) for p in pass_to_func]

        # zip_longest in case some stacks aren't found
        # wrapped function should raise an error if needed
        for fr, win in enumerate(itertools.zip_longest(*windows)):
            res = self.func(*calling_cls, *win, *args, **kwargs)
            if not fr:
                # If first frame, make the output array
                out = np.empty(out_shape, dtype=res.dtype)
            out[fr, ...] = res

        return out

    def _multiple_sliding_windows(self, arrs: Tuple[np.ndarray]) -> Generator:
        """
        Generates tuple with single frames from each np.ndarray
        """
        # Make generator for each window
        windows = []
        for arr in arrs:
            windows.append(sliding_window_generator(arr))

        # Yield from all generators simultaneously
        yield from zip(*windows)
