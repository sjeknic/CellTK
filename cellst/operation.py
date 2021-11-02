from typing import Collection, Tuple
import warnings

import numpy as np

from cellst.utils._types import Image, Mask, Track, Arr, INPT_NAME_IDX


class Operation():
    """
    This is the base class for the operations (segmentation, tracking, etc.)
    """
    __name__ = 'Operation'
    __slots__ = ('save', 'output', 'functions', 'func_index', 'input_images',
                 'input_masks', 'input_tracks', 'input_arrays', 'save_arrays',
                 'output_id', '_input_type', '_output_type')

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

        # Create a class to save intermediate arrays for saving
        self.save_arrays = {}

        # Create output id for tracking the images
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
                                  name: str = None,
                                  *args,
                                  **kwargs
                                  ) -> None:
        """
        args and kwargs are passed directly to the function.
        if name is not None, the files will be saved in a separate folder

        TODO:
            - Add option for func to be a Callable
            - Is func_index needed at all?
        """
        try:
            func = getattr(self, func)
            self.functions.append(tuple([func, output_type, args, kwargs, name]))
        except NameError:
            raise NameError(f"Function {func} not found in {self}.")

        self.func_index = {i: f for i, f in enumerate(self.functions)}

    def __call__(self,
                 images: Collection[Image] = [],
                 masks: Collection[Mask] = [],
                 tracks: Collection[Track] = [],
                 arrays: Collection[Arr] = []
                 ) -> (Image, Mask, Track, Arr):
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        Each BaseOperation should implement it's own __call__
        """
        return self.run_operation(images, masks, tracks, arrays)

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
        """
        # Default is to return the input if no function is run
        inputs = [images, masks, tracks, arrays]
        result = inputs[INPT_NAME_IDX[self._output_type.__name__]]

        for (func, expec_type, args, kwargs, name) in self.functions:
            output_type, result = func(*inputs, *args, **kwargs)

            # The user-defined expected type will overwrite output_type
            output_type = expec_type if expec_type is not None else output_type

            # Pass the result to the next function
            # TODO: This will currently raise a KeyError if it gets an unexpected type
            if isinstance(result, np.ndarray):
                inputs[INPT_NAME_IDX[output_type.__name__]] = [result]
            else:
                inputs[INPT_NAME_IDX[output_type.__name__]] = result

            # Save the function if needed for Pipeline to write files
            # By default, intermediate steps are saved in folder name
            # with file name output_type.__name__
            if name is not None:
                self.save_arrays[name] = output_type.__name__, result

        # Save the final result for saving as well
        # By default, final result is saved in folder self.output
        # with file name as self.output
        self.save_arrays[self.output] = self.output, result

        return result


class BaseProcess(Operation):
    __name__ = 'Process'
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

    def __call__(self,
                 images: Collection[Image] = [],
                 ) -> Image:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, [], [], [])


class BaseSegment(Operation):
    __name__ = 'Segment'
    _input_type = (Image,)
    _output_type = Mask

    def __init__(self,
                 input_images: Collection[str] = [],
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

    def __call__(self,
                 images: Collection[Image] = [],
                 masks: Collection[Mask] = []
                 ) -> Mask:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, masks, [], [])


class BaseTrack(Operation):
    __name__ = 'Track'
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

    def __call__(self,
                 images: Collection[Image] = [],
                 masks: Collection[Mask] = []
                 ) -> Track:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, masks, [], [])


# TODO: Write new run_operation to not pass arrays to the
#       extract data from image function
class BaseExtract(Operation):
    __name__ = 'Extract'
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
                 lineages: Collection[np.ndarray] = [],
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
            channels = input_images

        if len(regions) == 0:
            regions = input_masks

        kwargs = dict(channels=channels, regions=regions, lineages=lineages,
                      condition=condition)
        # Automatically add extract_data_from_image
        self.functions = [tuple([self.extract_data_from_image, Arr, [], kwargs, None])]
        self.func_index = {i: f for i, f in enumerate(self.functions)}

    def add_function_to_operation(self,
                                  func: str,
                                  output_type: type = None,
                                  name: str = None,
                                  *args,
                                  **kwargs
                                  ) -> None:
        """
        Extract currently only supports one function due to how
        extract_data_from_images expects the inputs. Therefore, for
        now there is no easy way to add a different function.

        TODO:
            - Implement option for new functions
        """
        raise NotImplementedError('Adding new functions to Extract is '
                                  'not currently supported. '
                                  'extract_data_from_image is included '
                                  'by default.')

    def __call__(self,
                 images: Collection[Image],
                 masks: Collection[Mask] = [],
                 tracks: Collection[Track] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 lineages: Collection[np.ndarray] = [],
                 condition: str = None,
                 ) -> Arr:
        """
        This directly calls extract_data_from_image
        instead of using run_operation
        """
        kwargs = dict(channels=channels, regions=regions,
                      condition=condition, lineages=lineages)
        return self.run_operation(images, masks, tracks, [], **kwargs)

    def run_operation(self,
                      images: Collection[np.ndarray] = [],
                      masks: Collection[np.ndarray] = [],
                      tracks: Collection[np.ndarray] = [],
                      arrays: Collection[np.ndarray] = [],
                      *args, **kwargs
                      ) -> (Image, Mask, Track, Arr):
        """
        Extract function is different because it expects to get a list
        of images, masks, etc. Therefore, can't use @ImageHelper.
        Instead, run_operation should directly call extract_data_from_image
        with the lists of inputs and return the result.
        Only note is that currently extract_data_from_images has no use
        for arrays, therefore those are not passed, but they must be input
        """
        # Default is to return the input if no function is run
        inputs = [images, masks, tracks, arrays]
        result = arrays

        # Arrays are not passed to the function
        result = self.extract_data_from_image(*inputs[:-1], *args, **kwargs)
        self.save_arrays[self.output] = self.output, result

        return result


class BaseEvaluate(Operation):
    __name__ = 'Evaluate'
    _input_type = (Arr,)
    _output_type = Arr

    def __call__(self,
                 arrs: Collection[Arr]
                 ) -> Arr:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation([], [], [], arrs)
