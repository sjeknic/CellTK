from typing import Collection, Tuple, Callable
import warnings

import numpy as np
from skimage.measure import regionprops_table

from cellst.utils._types import Image, Mask, Track, Arr, INPT_NAME_IDX
# TODO: For whole project. Probably move towards using modules more.
#       i.e. import cellst.utils.operation_utils as op
from cellst.utils.operation_utils import track_to_mask, parents_from_track


class Operation():
    """
    This is the base class for all operations (segmentation, tracking, etc.)
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
                 input_images: Collection[str] = [],
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


class BaseExtract(Operation):
    __name__ = 'Extract'
    _input_type = (Image, Mask, Track)
    _output_type = Arr
    # This is directly from skimage.regionprops
    _possible_metrics = ('area', 'bbox', 'bbox_area', 'centroid',
                         'convex_area', 'convex_image', 'coords',
                         'eccentricity', 'equivalent_diameter',
                         'euler_number', 'extent',
                         'feret_diameter_max', 'filled_area',
                         'filled_image', 'image',
                         'inertia_tensor', 'inertia_tensor_eigvals',
                         'intensity_image', 'label', 'local_centroid',
                         'major_axis_length', 'max_intensity',
                         'mean_intensity', 'min_intensity',
                         'minor_axis_length', 'moments',
                         'moments_central', 'moments_hu',
                         'moments_normalized', 'orientation', 'perimeter',
                         'perimeter_crofton', 'slice', 'solidity',
                         'weighted_centroid', 'weighted_local_centroid',
                         'weighted_moments', 'weighted_moments_hu',
                         'weighted_moments_normalized')
    # Label must always be first, even for user supplied metrics
    _metrics = ['label', 'area', 'convex_area',
                # 'equivalent_diameter_area', 'centroid_weighted',
                'mean_intensity',
                'orientation', 'perimeter', 'solidity',
                # TODO: centroid metrics and bbox could probably be included as scalars
                # 'bbox', 'centroid', 'coords'  # require multiple scalars
                ]
    _extra_properties = {}

    class EmptyProperty():
        """
        This class is to be used with skimage.regionprops_table.
        Every extra property passed to regionprops_table must
        have a unique name, however, I want to use several as a
        placeholder, so that I can get the right size array, but fill
        in the values later. So, this assigns a random __name__.
        """
        def __init__(self, *args) -> None:
            rng = np.random.default_rng()
            # Make it extremely unlikely to get the same int
            self.__name__ = str(rng.integers(999999))

        @staticmethod
        def __call__(empty):
            return np.nan

    def __init__(self,
                 input_images: Collection[str] = [],
                 input_masks: Collection[str] = [],
                 input_tracks: Collection[str] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 lineages: Collection[np.ndarray] = [],
                 condition: str = '',
                 min_trace_length: int = 0,
                 remove_parent: bool = True,
                 output: str = 'data_frame',
                 save: bool = True,
                 _output_id: Tuple[str] = None
                 ) -> None:
        """
        channel and region _map should be the names that will get saved in the final df
        with the images and masks they correspond to.
        TODO:
            - Clean up this whole function when it's working
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

        # Prefer tracks for naming
        if len(regions) == 0:
            if len(self.input_tracks) > 0:
                regions = input_tracks
            else:
                regions = input_masks

        # Name must be given
        if condition is None or condition == '':
            warnings.warn('Name of CellArray cannot be None or empty string.',
                          UserWarning)
            condition = 'default'

        # These kwargs get passed to self.extract_data_from_image
        kwargs = dict(channels=channels, regions=regions, lineages=lineages,
                      condition=condition, min_trace_length=min_trace_length,
                      remove_parent=remove_parent)
        # Automatically add extract_data_from_image
        # Name is always None, because gets saved in Pipeline as output
        self.functions = [tuple([self.extract_data_from_image, Arr, [], kwargs, None])]
        self.func_index = {i: f for i, f in enumerate(self.functions)}

        # Add division_frame and parent_id
        self._extra_properties.update(dict(division_frame=self.EmptyProperty(),
                                           parent_id=self.EmptyProperty()))

    def __call__(self,
                 images: Collection[Image],
                 masks: Collection[Mask] = [],
                 tracks: Collection[Track] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 lineages: Collection[np.ndarray] = [],
                 condition: str = None
                 ) -> Arr:
        """
        This directly calls extract_data_from_image
        instead of using run_operation
        """

        kwargs = dict(channels=channels, regions=regions,
                      condition=condition, lineages=lineages)
        return self.extract_data_from_image(images, masks, tracks,
                                            **kwargs)

    def _extract_data_with_track(self,
                                 image: Image,
                                 track: Track,
                                 metrics: Collection[str],
                                 extra_metrics: Collection[Callable],
                                 cell_index: dict = None
                                 ) -> np.ndarray:
        """
        Function

        Hard rule: parent must appear sequentially BEFORE daughter.
                   even the same frame won't work I think. But that
                   shouldn't be that hard to enforce

        NOTE: Final data structure has frames in last axis. In this
              function, frames is in first axis for faster np functions.
              np.moveaxis at the end to get correct structure. This is
              faster than writing to the last axis.
        """
        '''NOTE: cell_index should maybe be required arg. If calculated
        here, all other tracks in data set have to match or data will
        get overwritten / raise IndexError.'''
        if cell_index is None:
            cells = np.unique(track[track > 0])
            cell_index = {int(a): i for i, a in enumerate(cells)}

        # Get information about cell division
        daughter_to_parent = parents_from_track(track)
        mask = track_to_mask(track)

        # Organize metrics and get indices for custom ones
        all_metrics = metrics + list(self._extra_properties.keys())
        metric_idx = {k: i for i, k in enumerate(all_metrics)}

        # Build output
        out = np.empty((image.shape[0], len(all_metrics), len(cell_index)))
        # TODO: Add option to use different pad
        out[:] = np.nan

        for frame in range(image.shape[0]):
            # Extract metrics from each region in frame
            rp = regionprops_table(mask[frame], image[frame],
                                   properties=metrics,
                                   extra_properties=extra_metrics)
            # frame_data.shape is (len(metrics), len(cells))
            frame_data = np.row_stack(tuple(rp.values()))

            # Label is in the first position
            for n, lab in enumerate(frame_data[0, :]):
                # Cast to int for indexing
                lab = int(lab)

                if lab in daughter_to_parent:
                    # Get parent label
                    # NOTE: Could this ever raise a KeyError?
                    par = daughter_to_parent[lab]

                    # Copy parent trace to location of daughter trace
                    # Everything after frame is overwritten by daughter trace
                    out[:, :, cell_index[lab]] = out[:, :, cell_index[par]]

                    # Add division data to frame_data before saving
                    try:
                        frame_data[metric_idx['division_frame'], n] = frame
                    except KeyError:
                        pass

                    try:
                        frame_data[metric_idx['parent_id'], n] = par
                    except KeyError:
                        pass

                # Save frame data
                out[frame, :, cell_index[lab]] = frame_data[:, n]

        return np.moveaxis(out, 0, -1)

    def add_extra_metric(self, name: str, func: Callable = None) -> None:
        """
        Allows for adding custom metrics. If function is none, value will just
        be nan.
        """
        if name in self._metrics:
            warnings.warn(f'Metric {name} already exists.', UserWarning)
        else:
            if func is None: func = self.EmptyProperty()
            self._extra_properties[name] = func

    def set_metric_list(self, metrics: Collection[str]) -> None:
        """
        Adds metrics the user wants to pass to regionprops. Label will be made the
        first argument by default.
        """
        # Check that skimage can handle the given metrics
        allowed = [m for m in metrics if m in self._possible_metrics]
        not_allowed = [m for m in metrics if m not in self._possible_metrics]

        self._metrics = allowed

        # Raise warning for the rest
        if len(not_allowed) > 0:
            warnings.warn(f'Metrics {[not_allowed]} are not supported by skimage. '
                          'Use CellArray.add_extra_metric to add custom metrics.')

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
            - Implement option for new metrics during extract
        """
        raise NotImplementedError('Adding new functions to Extract is '
                                  'not currently supported. '
                                  'extract_data_from_image is included '
                                  'by default.')

    def run_operation(self,
                      images: Collection[np.ndarray] = [],
                      masks: Collection[np.ndarray] = [],
                      tracks: Collection[np.ndarray] = [],
                      arrays: Collection[np.ndarray] = [],
                      **kwargs
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
        inputs = [images, masks, tracks]

        # Get inputs that were saved during __init__
        _, expec_type, args, kwargs, name = self.functions[0]

        # Arrays are not passed to the function
        result = self.extract_data_from_image(*inputs, *args, **kwargs)
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
