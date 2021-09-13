import types
import inspect
from typing import NewType

import numpy as np

# Define custom types to make output tracking esier
Image = NewType('image', np.ndarray)
Mask = NewType('mask', np.ndarray)
Track = NewType('track', np.ndarray)

INPT_NAMES = [Image.__name__, Mask.__name__, Track.__name__]
INPT_NAME_IDX = {n: i for i, n in enumerate(INPT_NAMES)}
INPT = [Image, Mask, Track]
INPT_IDX = {n: i for i, n in enumerate(INPT)}

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


# Decorator to help with passing only the correct images to functions
def image_helper(func):
    """
    Passes only the required image stacks to the function
    Returns the expected output type as well as the result
    """
    # Determine the expected output type
    output_type = inspect.signature(func).return_annotation.__name__

    def decorator(imgs, msks, trks, *args, **kwargs):
        # TODO: Change this to be based on type.
        # Check which inputs the function is expecting and only pass those
        expected = [p.name for p in inspect.signature(func).parameters.values()]
        inpt_bools = [i in expected for i in INPT_NAMES]
        pass_to_func = [i for b, inpt in zip(inpt_bools, [imgs, msks, trks])
                        for i in inpt if b]

        # Because this function is expecting a stack, it should always return a stack
        stack = func(*pass_to_func, *args, **kwargs)
        # NOTE: won't work for 1D images. Does that matter?
        while stack.ndim < 3:
            stack = np.expand_dims(stack, axis=-1)

        return output_type, stack
    return decorator
