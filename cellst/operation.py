from typing import Collection, Tuple

import numpy as np

from cellst.utils._types import Image, Mask, Track, Arr, INPT_NAME_IDX


class Operation():
    """
    This is the base class for the operations (segmentation, tracking, etc.)

    TODO:
        - Implement __slots__
    """

    def __init__(self,
                 output: str,
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 **kwargs
                 ) -> None:
        """
        by default only the last output can be saved (should this change?)

        TODO:
            - Add more name options, like a specific output folder name
            - Outputs are now a required arg for the base class.
            - Add name or other easy identifier for error messages
        """
        self.save = save
        self.output = output

        # These are used to track what the operation has been asked to do
        self.functions = []
        self.func_index = {}

        # Will be overwritten by inheriting class if they are used
        # Otherwise, these defaults can be used to know if they haven't
        self.input_images = []
        self.input_masks = []
        self.input_tracks = []
        self.input_arrays = []

        if _output_id is not None:
            self.output_id = _output_id
        else:
            output_type = self._output_type.__name__
            self.output_id = tuple([output, output_type])

    def __setattr__(self, name, value) -> None:
        '''TODO: Not needed here, but the idea behind a custom __setattr__
               class is that the inheriting Operation can decide if the function
               meets the requirements.'''
        super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        Returns printable version of the functions and args in Operation

        TODO:
            - Return function name instead of decorator name
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
            - Add option for func to be a Callable
            - Is func_index needed at all?
        """
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
                      arrays: Collection[np.ndarray] = []
                      ) -> (Image, Mask, Track, Arr):
        """
        Rules for operation functions:
            Must take in at least one of image, mask, track
            Can take in as many of each, but must be a separate positional argument
            Either name or type hint must match the types above
            If multiple, must be present in above order

        TODO:
            - np array could be preallocated for the function
        """
        # Default is to return the input if no function is run
        inputs = [images, masks, tracks, arrays]
        result = inputs[INPT_NAME_IDX[self._output_type.__name__]]

        for (func, expec_type, args, kwargs) in self.functions:
            # TODO: Only gets output type from decorator or if
            #       function explicitly returns it
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


class BaseProcess(Operation):
    _input_type = (Image,)
    _output_type = Image

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 output: str = 'process',
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        self.output = output


class BaseSegment(Operation):
    _input_type = (Image,)
    _output_type = Mask

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 input_masks: Collection[str] = [],
                 output: str = 'mask',
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        if isinstance(input_masks, str):
            self.input_masks = [input_masks]
        else:
            self.input_masks = input_masks

        self.output = output


class BaseTrack(Operation):
    _input_type = (Image, Mask)
    _output_type = Track

    def __init__(self,
                 input_images: Collection[str] = [],
                 input_masks: Collection[str] = [],
                 output: str = 'track',
                 save: bool = False,
                 track_file: bool = True,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        if isinstance(input_masks, str):
            self.input_masks = [input_masks]
        else:
            self.input_masks = input_masks

        self.output = output


class BaseExtract(Operation):
    _input_type = (Image, Mask, Track)
    _output_type = Arr
    # Label will always be first, even for user supplied metrics
    """TODO: This is important. xr.DataArray can only handle a single datatype.
    This is similar to CellTK/covertrace. However, it prevents the use of some interesting
    properties, such as coords. An approach to fix this is to use an xr.DataSet to hold multiple
    xr.DataArrays of multiple types. However, I am concerned that this will
    greatly slow down the indexing, and it also complicates returning multiple
    values. For now, I'm just going to focus on metrics that can be stored as scalars.
    TODO: Additionally, some of the metrics that can be stored as scalars are returned
    as multiple metrics (i.e. bbox = bbox-0, bbox-1, bbox-2, bbox-3). These need to be
    automatically handled. See scipy.regionprops and scipy.regionprops_table for
    more information on which properites."""
    _metrics = ['label', 'area', 'convex_area',
                # 'equivalent_diameter_area',
                # 'centroid_weighted',  # centroid weighted by intensity
                'mean_intensity',
                # 'intensity_mean', 'intensity_min',  # need to add median intensity
                'orientation', 'perimeter', 'solidity',
                # 'bbox', 'centroid', 'coords'  # require multiple scalars
                ]
    _extra_properties = []

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 input_masks: Collection[str] = [],
                 input_tracks: Collection[str] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 condition: str = None,
                 output: str = 'data_frame',
                 save: bool = False,
                 _output_id: Tuple[str] = None
                 ) -> None:
        """
        channel and region _map should be the names that will get saved in the final df
        with the images and masks they correspond to.
        TODO:
            - Clean up this whole function when it's working
            - metrics needs to always start with label
            - If Mask is given, but not track, look for tracking file
                - Raise warning that parent daughter connections will fail if missing
            - Condition should get passed to this function. Pipeline likely cannot because
              the same operation will be used for multiple Pipelines. But Orchestrator should
              be able to.
        """

        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        if isinstance(input_masks, str):
            self.input_masks = [input_masks]
        else:
            self.input_masks = input_masks

        if isinstance(input_tracks, str):
            self.input_tracks = [input_tracks]
        else:
            self.input_tracks = input_tracks

        if len(channels) == 0:
            self.channels = input_images
        else:
            self.channels = channels

        # TODO: This should look for both masks and tracks
        if len(regions) == 0:
            self.regions = input_masks
        else:
            self.regions = regions

        # Only one function can be used for the Extract module
        # TODO: Overwrite add_function to raise an error, user was
        #       likely trying to add extra_properties instead
        self.functions = [tuple([self.extract_data_from_image, Arr, [], {}])]
        self.func_index = {i: f for i, f in enumerate(self.functions)}


class BaseEvaluate(Operation):
    _input_type = (Arr,)
    _output_type = Arr
