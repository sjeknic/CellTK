import warnings
from typing import Collection, Tuple, Callable, List, Dict
import logging
import time

import numpy as np
from skimage.measure import regionprops_table

from cellst.utils._types import Image, Mask, Track, Arr, INPT_NAME_IDX
from cellst.utils.operation_utils import (track_to_mask, parents_from_track,
                                          RandomNameProperty)
from cellst.utils.log_utils import get_console_logger
import cellst.utils.metric_utils as metric_utils


class Operation():
    __name__ = 'Operation'
    __slots__ = ('save', 'output', 'functions', 'func_index', 'input_images',
                 'input_masks', 'input_tracks', 'input_arrays', 'save_arrays',
                 'output_id', '_input_type', '_output_type', 'logger', 'timer')

    def __init__(self,
                 output: str,
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 **kwargs
                 ) -> None:
        """

        """
        # Get empty logger, will be overwritten by Pipeline
        self.logger = get_console_logger()

        # Save basic params
        self.save = save
        self.output = output

        # These are used to track what the operation has been asked to do
        self.functions = []
        self.func_index = {}

        # Will be overwritten by inheriting class if they are used
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
            try:
                output_type = self._output_type.__name__
            except AttributeError:
                output_type = None
            self.output_id = tuple([output, output_type])

    def __str__(self) -> str:
        """Returns printable version of the functions and args in Operation."""
        op_id = f'{self.__name__} at {hex(id(self))}'

        # Get the names of the inputs
        inputs = tuple([f"{name[0]}:{getattr(self, f'input_{name}s')}"
                        for name in INPT_NAME_IDX.keys()
                        if getattr(self, f'input_{name}s')])

        # Format each function as a str
        funcs = []
        for (func, otpt, args, kwargs, name) in self.functions:
            funcs.append(self._format_function_string(func, args, kwargs))
        fstr = ' -> '.join(funcs)

        # Get the name of the output
        output = f'{self.output_id[1]}:{self.output_id[0]}'

        # Put it all together
        string = f"{op_id}: {inputs} -> {fstr} -> {output}"

        return string

    def __call__(self,
                 images: Collection[Image] = [],
                 masks: Collection[Mask] = [],
                 tracks: Collection[Track] = [],
                 arrays: Collection[Arr] = []
                 ) -> (Image, Mask, Track, Arr):
        """
        __call__ runs operation independently of Pipeline class
        """
        return self.run_operation(images, masks, tracks, arrays)

    def __enter__(self) -> None:
        """
        """
        if not hasattr(self, 'save_arrays'):
            self.save_arrays = {}

        # Start a timer
        self.timer = time.time()

        # Log relevant information (would not have been logged in init)
        self.logger.info(f'Operation {self.__name__} at '
                         f'{hex(id(self))} entered.')
        # Log requests for each data type
        for name in INPT_NAME_IDX.keys():
            if getattr(self, f'input_{name}s'):
                self.logger.info(f"input_{name}:{getattr(self, f'input_{name}s')}")
        self.logger.info(f"Output ID: {self.output_id}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        """
        # Delete save arrays in memory
        self.save_arrays = None
        del self.save_arrays

        # Log time spent after enter
        try:
            self.logger.info(f'{self.__name__} execution time: '
                             f'{time.time() - self.timer}')
            self.timer = None
        except TypeError:
            # KeyboardInterrupt now won't cause additional exceptions
            pass

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
            # Check that the function exists
            _ = getattr(self, func)

            # Save func as string for dictionaries
            self.functions.append(tuple([func, output_type, args, kwargs, name]))
        except AttributeError:
            raise AttributeError(f"Function {func} not found in {self}.")

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
            func = getattr(self, func)

            # Run the function
            self.logger.info(f'Starting function {func.__name__}')
            self.logger.info(f'Inputs: {[(i.shape, i.dtype) for inpt in inputs for i in inpt]}')
            exec_timer = time.time()
            output_type, result = func(*inputs, *args, **kwargs)

            # The user-defined expected type will overwrite output_type
            output_type = expec_type if expec_type is not None else output_type
            self.logger.info(f'Output: {output_type.__name__}, {result.shape}, '
                             f'{result.dtype}')
            self.logger.info(f'{func.__name__} execution time: '
                             f'{time.time() - exec_timer}.')

            # Pass the result to next func - raises KeyError on unexpected type
            if isinstance(result, np.ndarray):
                inputs[INPT_NAME_IDX[output_type.__name__]] = [result]
            else:
                inputs[INPT_NAME_IDX[output_type.__name__]] = result

            # Save the function if needed for Pipeline to write files
            if name is not None:
                self.save_arrays[name] = output_type.__name__, result

        # Save the final result for saving as well
        self.save_arrays[self.output] = self.output, result

        return result

    def set_logger(self, logger: logging.Logger) -> None:
        """
        """
        # logger is either a Pipeline or Operation logger
        log_name = logger.name

        # This logs to same file, but records the Operation name
        self.logger = logging.getLogger(f'{log_name}.{self.__name__}')

    def _operation_to_dict(self, op_slots: Collection[str] = None) -> Dict:
        """
        Returns a dictionary that fully defines the operation
        """
        # Get attributes to lookup
        base_slots = ['__name__', '__module__', 'save', 'output', '_output_id']
        if op_slots is not None: base_slots.extend(op_slots)

        # Save in dictionary
        op_defs = {}
        for att in base_slots:
            op_defs[att] = getattr(self, att, None)

        # Save function definitions
        func_defs = {}
        for func, output_type, args, kwargs, name in self.functions:
            func_defs[func] = {}
            if output_type is not None:
                func_defs[func]['output_type'] = output_type.__name__
            else:
                func_defs[func]['output_type'] = output_type
            func_defs[func]['name'] = name
            func_defs[func]['args'] = args
            func_defs[func]['kwargs'] = kwargs

        # Save in original dictionary
        op_defs['_functions'] = func_defs

        return op_defs

    def _format_function_string(self,
                                fname: str,
                                args: tuple,
                                kwargs: dict
                                ) -> str:
        """
        Nicely formats the function specifications for the Operation

        TODO: is there a way to neatly include type and save name?
        """
        # Format args and kwargs to str
        if len(args) > 0:
            str_args = ', '.join(tuple([str(a) for a in args]))
        else:
            str_args = ''

        if len(kwargs) > 0:
            str_kwargs = ', '.join(tuple([f'{k}={v}'
                                          for k, v in kwargs.items()]))
        else:
            str_kwargs = ''

        # Format the arg strings nicely
        if not str_args and not str_kwargs:
            passed = ''
        elif not str_args and str_kwargs:
            passed = str_kwargs
        elif str_args and not str_kwargs:
            passed = str_args
        else:
            passed = f'{str_args}, {str_kwargs}'

        return f'{fname}({passed})'


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

    def _operation_to_dict(self) -> Dict:
        op_slots = ['input_images']
        return super()._operation_to_dict(op_slots)


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

    def _operation_to_dict(self) -> Dict:
        op_slots = ['input_images', 'input_masks']
        return super()._operation_to_dict(op_slots)


class BaseTrack(Operation):
    __name__ = 'Track'
    _input_type = (Image, Mask)
    _output_type = Track

    def __init__(self,
                 input_images: Collection[str] = [],
                 input_masks: Collection[str] = [],
                 output: str = 'track',
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
                 ) -> Track:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, masks, [], [])

    def _operation_to_dict(self) -> Dict:
        op_slots = ['input_images', 'input_masks']
        return super()._operation_to_dict(op_slots)


class BaseExtract(Operation):
    """TODO: Include add_derived_metrics"""
    __name__ = 'Extract'
    _input_type = (Image, Mask, Track)
    _output_type = Arr
    # This is directly from skimage.regionprops
    # Not included yet: convex_image, filled_image, image, inertia_tensor
    #                   inertia_tensor_eigvals, local_centroid, intensity_image
    #                   moments, moments_central, moments_hu, coords,
    #                   moments_normalized, slice, weighted_centroid,
    #                   weighted_local_centroid, weighted_moments,
    #                   weighted_moments_hu, weighted_moments_normalized
    _possible_metrics = ('area', 'bbox', 'bbox_area', 'centroid',
                         'convex_area',
                         'eccentricity', 'equivalent_diameter',
                         'euler_number', 'extent',
                         'feret_diameter_max', 'filled_area', 'label',
                         'major_axis_length', 'max_intensity',
                         'mean_intensity', 'min_intensity',
                         'minor_axis_length', 'major_axis_length',
                         'orientation', 'perimeter',
                         'perimeter_crofton', 'solidity')

    _metrics = ['label', 'area', 'convex_area', 'filled_area', 'bbox',
                'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                'minor_axis_length', 'major_axis_length',
                'orientation', 'perimeter', 'solidity']
    _extra_properties = ['division_frame', 'parent_id', 'total_intensity',
                         'median_intensity']
    _props_to_add = {}

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
        channels and regions should be the names that will get saved in the final df
        with the images and masks they correspond to.
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
        self.functions = [tuple(['extract_data_from_image', Arr, [], kwargs, None])]
        self.func_index = {i: f for i, f in enumerate(self.functions)}

        # Add division_frame and parent_id
        for m in self._extra_properties:
            self.add_extra_metric(m)

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

    @property
    def metrics(self) -> list:
        return self._metrics + list(self._props_to_add.keys())

    @property
    def extract_kwargs(self) -> dict:
        return self.functions[0][3]  # index for kwargs passed to extract_data

    def _correct_metric_dim(self, met_list: List[str]) -> List[str]:
        """
        Adjust all measures for the presence of multi-scalar metrics
        """
        new_met_list = []
        for m in met_list:
            if m == 'bbox':
                # bbox becomes bbox-1,...,bbox-4'
                new_met_list.extend([f'bbox-{n}' for n in range(1, 5)])
            elif m == 'centroid':
                # centroid becomes centroid-1, centroid-2
                new_met_list.extend([f'centroid-{n}' for n in range(1, 3)])
            else:
                new_met_list.append(m)

        return new_met_list

    def _extract_data_with_track(self,
                                 image: Image,
                                 track: Track,
                                 metrics: Collection[str],
                                 extra_metrics: Collection[Callable],
                                 cell_index: dict = None
                                 ) -> np.ndarray:
        """
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
        all_metrics = metrics + list(self._props_to_add.keys())
        all_metrics = self._correct_metric_dim(all_metrics)
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

    def _operation_to_dict(self) -> Dict:
        op_slots = ['input_images', 'input_masks', 'input_tracks']
        op_dict = super()._operation_to_dict(op_slots)

        # Add the kwargs for extract_data_from_image
        op_dict.update(self.extract_kwargs)

        # Add metrics and extra properties
        # TODO: This is also a bit hackish
        func = 'extract_data_from_image'
        op_dict['_functions'][func]['metrics'] = self._metrics
        op_dict['_functions'][func]['extra_props'] = self._props_to_add

        return op_dict

    def add_extra_metric(self, name: str, func: Callable = None) -> None:
        """
        Allows for adding custom metrics. If function is none, value will just
        be nan.
        """
        if func is None:
            if name in self._possible_metrics:
                self._metrics.append(name)
            else:
                try:
                    func = getattr(metric_utils, name)
                except AttributeError:
                    # Function not implemented by me
                    func = RandomNameProperty()

        self._props_to_add[name] = func

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
        extract_data_from_images expects the inputs.
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
        """
        # Arrays are not passed to extract_data_from_image
        inputs = [images, masks, tracks]

        # Get inputs that were saved during __init__
        _, expec_type, args, kwargs, name = self.functions[0]

        # Extract needs separate logging here
        self.logger.info('Starting function extract_data_from_image')
        self.logger.info(f"Channels: {list(zip(kwargs['channels'], self.input_images))}")
        self.logger.info(f"Regions: {list(zip(kwargs['regions'], self.input_tracks + self.input_masks))}")
        self.logger.info(f"Metrics: {self._metrics}")
        self.logger.info(f"Added metrics: {list(self._props_to_add.keys())}")
        self.logger.info(f"Condition: {kwargs['condition']}")

        # Run function
        exec_timer = time.time()
        result = self.extract_data_from_image(*inputs, *args, **kwargs)
        self.save_arrays[self.output] = self.output, result

        # Log results
        self.logger.info(f'Output: {result.name}, {result.shape}, '
                         f'{result.dtype}')
        self.logger.info(f'extract_data_from_image execution time: '
                         f'{time.time() - exec_timer}.')

        return result


class BaseEvaluate(Operation):
    __name__ = 'Evaluate'
    _input_type = (Arr,)
    _output_type = Arr

    def __init__(self,
                 input_arrays: Collection[str] = [],
                 output: str = 'evaluate',
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_arrays, str):
            self.input_arrays = [input_arrays]
        else:
            self.input_arrays = input_arrays

    def __call__(self,
                 arrs: Collection[Arr]
                 ) -> Arr:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation([], [], [], arrs)

    def _operation_to_dict(self) -> Dict:
        op_slots = ['input_arrays']
        return super()._operation_to_dict(op_slots)
