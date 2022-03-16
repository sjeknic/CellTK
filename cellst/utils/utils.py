import sys
import os
import functools
import itertools
import contextlib
import warnings
import logging
import typing
from copy import deepcopy
from itertools import zip_longest
from typing import Tuple, Generator, Dict
from collections import Counter

import numpy as np

from cellst.utils._types import (Image, Mask, Track,
                                 ImageContainer, INPT_NAMES)
from cellst.core.arrays import ConditionArray, ExperimentArray
from cellst.utils.operation_utils import sliding_window_generator
from cellst.utils.log_utils import get_null_logger


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
        - Pass default args if no user input. See:
        https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
        - Remove self.expected_types, self.expected_names, self.expected_numbers
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

        # NOTE: self/cls is not included in these type hints
        self.type_hints = typing.get_type_hints(self.func)
        try:
            # Save as type object, not __name__
            self.output_type = self.type_hints.pop('return')
            # Get the arguments the function expects
            self.expected_names, self.expected_types = zip(
                *[(k, v.__name__) for k, v in self.type_hints.items()
                  if hasattr(v, '__name__') and v.__name__ in INPT_NAMES]
            )
            # Update the number of each Stack type that is expected
            self.expected_numbers = dict(Counter(self.expected_types))
            self.expected_numbers.update({k: 0 for k in INPT_NAMES
                                          if k not in self.expected_types})
        except KeyError:
            raise KeyError(f'Type hints are missing for function {self.func}')

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            # Sort args and kwargs into the respective values
            if isinstance(args[0], ImageContainer):
                # If first value is not self or class, then staticmethod
                img_container = deepcopy(args[0])
                calling_cls = []
            else:
                calling_cls = [args[0]]
                img_container = deepcopy(args[1])

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
                stack = self.func(*calling_cls, **pass_to_func, **nkwargs)
            else:
                # Pass inputs individually
                stack = self._pass_by_frames(pass_to_func, calling_cls,
                                             **nkwargs)

            # Get the correct outputs and keys before returning
            keys, stack = self._correct_outputs(keys, stack, calling_cls)

            return keys, stack
        return wrapper

    def _type_helper(self, img_container, **kwargs):
        """
        This func is for sorting the input types and selecting the correct
        types that should get passed to the function.
        """
        # Check for what the function expects. Include plurals for the name
        inpt_bools = [(i in self.expected_types) for i in INPT_NAMES]
        self.logger.info('Function accepts: '
                         f'{[i for i, b in zip(INPT_NAMES, inpt_bools) if b]}')

        # Assign images from img_container to kwargs
        kwargs = self._name_helper(img_container, **kwargs)

        # Sort stacks into pass_to_func and save expected keys/sizes/types
        pass_to_func = {}
        keys = []
        inpt_size_type = []
        for kw, val in kwargs.items():
            # Check that this kw is for a Stack
            try:
                assert self.type_hints[kw].__name__ in INPT_NAMES
            except (AttributeError, AssertionError):
                # Indicates not CellST type
                continue

            if self.as_tuple:
                # No trimming of inputs is done if as_tuple
                if val:
                    key, stack = zip(*val)
                    pass_to_func[kw] = tuple(stack)
                    keys.extend(key)
                    inpt_size_type.extend([(s.shape, s.dtype)
                                           for s in stack])
                else:
                    # NOTE: This will overwrite default args. see TODO
                    pass_to_func[kw] = tuple([])
            else:
                if val:
                    # Pass only one stack, last one in list is newest
                    key, stack = val.pop()
                    pass_to_func[kw] = stack
                    keys.append(key)
                    inpt_size_type.append((stack.shape, stack.dtype))

        self.logger.info(f'Selected inputs: {list(zip(keys, inpt_size_type))} '
                         f'for kwargs: {list(pass_to_func.keys())}')
        # Remove inputs that are in pass_to_func so they don't get passed twice
        kwargs = {k: v for k, v in kwargs.items() if k not in pass_to_func}
        return keys, pass_to_func, kwargs

    def _name_helper(self, img_container, **kwargs):
        """
        Finds images the user specified using kwargs
        """
        # Load any names the user supplied in kwargs
        kwargs_to_load = [(n, t) for n, t in zip(self.expected_names,
                                                 self.expected_types)
                          if t in INPT_NAMES and n in kwargs]
        for kw, exp_typ in kwargs_to_load:
            # Get the user input
            names = kwargs[kw]

            if names is None:
                # User does not want any image passed
                names = []
            elif isinstance(names, str):
                # Append the expected type to the name
                names = [(names, exp_typ)]
            else:
                # First image given is first image passed
                names = [(nm, exp_typ) for nm in names]
                if not self.as_tuple:
                    warnings.warn('Multiple stacks for a single keyword '
                                  f'argument is not supported for {self.func}.'
                                  f'Only {names[0]} will get passed.',
                                  UserWarning)

            # Load the stacks from img_container and add to kwargs
            try:
                # Keep key with image for tracking purposes
                kwargs[kw] = []
                for n in names:
                    kwargs[kw].append((n, img_container.pop(n)))
            except KeyError:
                raise KeyError(f'Could not find input {n}.')

        # For unspecified args, load a single list for each type
        # Oldest stack is first, newest is last
        img_lists = {i: [(k, v) for k, v in img_container.items()
                         if k[1] == i]
                     for i in INPT_NAMES}

        # Load stack lists to kwargs
        for exp_name, exp_typ in zip(self.expected_names, self.expected_types):
            if exp_typ in INPT_NAMES and exp_name not in kwargs:
                kwargs[exp_name] = img_lists[exp_typ]

        return kwargs

    def _correct_outputs(self, keys, stack, calling_cls=[]):
        """Ensures keys and stack match and are in expected format"""
        # Store keys as list if not already
        if isinstance(keys[0], str):
            keys = [keys]

        # If output_type is "same" out_type = in_type, else defined by function
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
            # TODO: This doesn't work with Same type output yet
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
            raise ValueError(f'Length of outputs ({len(stack)}) longer '
                             f'than length of keys ({len(keys)}).')

        # Adjust array dimensions if needed
        for n, (k, st) in enumerate(zip(keys, stack)):
            while st.ndim < 3 and any([i in k for i in (Image, Mask, Track)]):
                st = np.expand_dims(st, axis=-1)
            stack[n] = st

        return keys, stack

    def _get_calling_logger(self, calling_cls):
        """
        Gets the operation logger to record info about inputs and outputs
        """
        if calling_cls:
            try:
                logger = calling_cls[0].logger
                return logging.getLogger(f'{logger.name}.{self.__name__}')
            except AttributeError:
                return get_null_logger()

    def _pass_by_frames(self,
                        pass_to_func: Dict,
                        calling_cls: Tuple,
                        **kwargs
                        ) -> np.ndarray:
        """
        Creates output array and passes individual frames to func

        TODO:
            - Is it okay to convert float64 -> float32? or int32 -> int16
        """
        # Get window generators for each array in pass_to_func
        if self.as_tuple:
            out_shape = next(iter(pass_to_func.values()))[0].shape
            windows = {p: self._multiple_sliding_windows(v)
                       for p, v in pass_to_func.items()}
            _fv = tuple([])
        else:
            out_shape = next(iter(pass_to_func.values())).shape
            windows = {p: sliding_window_generator(v)
                       for p, v in pass_to_func.items()}
            _fv = None

        # zip_longest in case some stacks aren't found
        # wrapped function should raise an error if this is a problem
        for fr, win in enumerate(itertools.zip_longest(*windows.values(), fillvalue=_fv)):
            new_win = dict(zip(pass_to_func.keys(), win))
            res = self.func(*calling_cls, **new_win, **kwargs)
            if not fr:
                # If first frame, make the output array
                out_shape = (out_shape[0], *res.shape)
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
