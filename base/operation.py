from typing import Collection

import numpy as np
import skimage.measure as meas

from base.utils import image_helper, Image, Mask, Track, INPT_NAME_IDX


class Operation(object):
    """
    This is the base class for the operations (segmentation, tracking, etc.)
    """

    def __init__(self,
                 save: bool = False) -> None:
        """
        by default only the last output can be saved (should this change?)

        TODO:
            - Add more name options, like a specific output folder name
        """
        self.save = save

        self.functions = []
        self.func_index = {}

        # Will be overwritten by inheriting class if they are used
        self.input_images = []
        self.input_masks = []
        self.input_tracks = []
        self.output = None

    def __setattr__(self, name, value) -> None:
        '''TODO: Not needed here, but the idea behind a custom __setattr__
               class is that the inheriting Operation can decide if the function
               meets the requirements.'''
        super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        Returns printable version of the functions and args in Operation
        """
        string = str(super().__repr__())

        for k, v in self.func_index.items():
            string += (f'\nIndex {k}: \n'
                       f'Function: {v[0]} \n'
                       f'   args: {v[1]} \n'
                       f'   kwargs: {v[2]}')
        return string

    def add_function_to_operation(self,
                                  func: str,
                                  output_type: str = None,
                                  index: int = -1,
                                  *args,
                                  **kwargs
                                  ) -> None:
        """
        args and kwargs should be passed to the function.

        TODO:
            - Update output to no longer be str, but type
        """
        output_type = self._output_type if output_type is None else output_type

        if not hasattr(self, func):
            raise NameError(f"Function {func} not found in {self}.")
        else:
            func = getattr(self, func)
            self.functions.append(tuple([func, output_type, args, kwargs]))

        self.func_index = {i: f for i, f in enumerate(self.functions)}

    def run_operation(self,
                      images: Collection[np.ndarray] = [],
                      masks: Collection[np.ndarray] = [],
                      tracks: Collection[np.ndarray] = [],
                      ) -> (Image, Mask, Track):
        """
        Rules for operation functions:
            Must take in at least one of image, mask, track
            Can take in as many of each, but must be a separate positional argument
            Cannot be named anything other than that
            If multiple, must be present in above order
        """
        # Default is to return the input if no function is run
        inputs = [images, masks, tracks]
        result = inputs[INPT_NAME_IDX[self._output_type]]

        for (func, expec_type, args, kwargs) in self.functions:
            output_type, result = func(*inputs, *args, **kwargs)

            # The user-defined expected type will overwrite output_type
            output_type = expec_type if expec_type is not None else output_type

            # Pass the result to the next function
            # TODO: This will currently raise a KeyError if it gets an unexpected type
            if isinstance(result, np.ndarray):
                inputs[INPT_NAME_IDX[output_type]] = [result]
            else:
                inputs[INPT_NAME_IDX[output_type]] = result

        return result


class Preprocess(Operation):
    _input_type = ('image',)
    _output_type = 'image'
    pass


class Segment(Operation):
    _input_type = ('image',)
    _output_type = 'mask'

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 output_name: str = 'mask',
                 save: bool = False
                 ) -> None:
        super().__init__(save)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        self.output = output_name

    @staticmethod
    @image_helper
    def constant_thres(image: np.ndarray,
                       THRES=1000,
                       NEG=False
                       ) -> Image:
        if NEG:
            return meas.label(image < THRES)
        return meas.label(image > THRES)


class Track(Operation):
    _input_type = ('image', 'mask')
    _output_type = 'track'
    pass


class Postprocess(Operation):
    pass


class Evaluate(Operation):
    pass


class Extract(Operation):
    pass
