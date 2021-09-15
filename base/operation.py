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
        string = str(super().__str__())

        for k, v in self.func_index.items():
            string += (f'\nIndex {k}: \n'
                       f'Function: {v[0]} \n'
                       f'   args: {v[1]} \n'
                       f'   kwargs: {v[2]}')
        return string

    def add_function_to_operation(self,
                                  func: str,
                                  output_type: type = None,
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

        TODO:
            - np array could be preallocated for the function
        """
        # Default is to return the input if no function is run
        inputs = [images, masks, tracks]
        result = inputs[INPT_NAME_IDX[self._output_type.__name__]]

        for (func, expec_type, args, kwargs) in self.functions:
            output_type, result = func(*inputs, *args, **kwargs)

            # The user-defined expected type will overwrite output_type
            output_type = expec_type if expec_type is not None else output_type

            # Pass the result to the next function
            # TODO: This will currently raise a KeyError if it gets an unexpected type
            if isinstance(result, np.ndarray):
                inputs[INPT_NAME_IDX[output_type.__name__]] = [result]
            else:
                inputs[INPT_NAME_IDX[output_type.__name__]] = result

        return result


class Preprocess(Operation):
    _input_type = (Image,)
    _output_type = Image
    pass


class Segment(Operation):
    _input_type = (Image,)
    _output_type = Mask

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

    # TODO: Should these methods actually be static? What's the benefit?
    @staticmethod
    @image_helper
    def constant_thres(image: Image,
                       THRES=1000,
                       NEG=False
                       ) -> Image:
        if NEG:
            return meas.label(image < THRES)
        return meas.label(image > THRES)

    @image_helper
    def unet_predict(self,
                     image: Image,
                     weight_path: str,
                     roi: (int, str) = 2,
                     batch: int = None,
                     classes: int = 3,
                     ) -> Image:
        """
        NOTE: If we had mulitple colors, then image would be 4D here. The Pipeline isn't
        set up for that now, so for now the channels is just assumed to be 1.

        roi - the prediction values are returned only for the roi
        batch - number of frames passed to model. None is all of them.
        classes - number of output categories from the model (has to match weights)
        """
        # probably don't need as many options here
        _roi_dict = {'background': 0, 'bg': 0, 'edge': 1,
                     'interior': 2, 'nuc': 2, 'cyto': 2}
        if isinstance(roi, str):
            try:
                roi = _roi_dict[roi]
            except KeyError:
                raise ValueError(f'Did not understand region of interest {roi}.')

        # Only import tensorflow and Keras if needed
        from base.unet_utils import unet_model

        if not hasattr(self, 'model'):
            '''NOTE: If we had mulitple colors, then image would be 4D here.
            The Pipeline isn't set up for that now, so for now the channels
            is just assumed to be 1.'''
            channels = 1
            dims = (image.shape[1], image.shape[2], channels)
            self.model = unet_model.UNetModel(dimensions=dims,
                                              weight_path=weight_path,
                                              model='unet')

        # Pre-allocate output memory
        # TODO: Incorporate the batch here.
        output = np.empty(image.shape)
        for i in range(image.shape[0]):
            output[i, ...] = self.model.predict(image[i, ...])

        return output


class Track(Operation):
    _input_type = (Image, Mask)
    _output_type = Track
    pass


class Postprocess(Operation):
    pass


class Evaluate(Operation):
    pass


class Extract(Operation):
    pass
