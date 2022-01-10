import os
import warnings
from typing import Collection, Tuple, Callable, List, Dict
import logging
import time
import inspect

import numpy as np
from skimage.measure import regionprops_table

from cellst.utils._types import (Image, Mask, Track, Arr,
                                 ImageContainer, INPT_NAMES)
from cellst.utils.operation_utils import (track_to_mask, parents_from_track,
                                          RandomNameProperty)
from cellst.utils.log_utils import get_console_logger
import cellst.utils.metric_utils as metric_utils


class Operation():
    __name__ = 'Operation'
    __slots__ = ('save', 'output', 'functions', 'func_index', 'input_images',
                 'input_masks', 'input_tracks', 'input_arrays', 'save_arrays',
                 'output_id', 'logger', 'timer', '_split_key', 'force_rerun',
                 '_input_type', '_output_type')

    def __init__(self,
                 output: str,
                 save: bool = False,
                 force_rerun: bool = False,
                 _output_id: Tuple[str] = None,
                 _split_key: str = '&',
                 **kwargs
                 ) -> None:
        """
        """
        # Get empty logger, will be overwritten by Pipeline
        self.logger = get_console_logger()

        # Save basic params
        self.save = save
        self.output = output
        self.force_rerun = force_rerun
        self._split_key = _split_key

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
                        for name in INPT_NAMES
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

        TODO:
            - In order for __call__ to work with the new ImageContainer system
              it will have to take the inputs and build the ImageContainer. Only
              question will be the keys, they have to match the inputs.
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
        for name in INPT_NAMES:
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
                                  save_name: str = None,
                                  *args,
                                  **kwargs
                                  ) -> None:
        """
        args and kwargs are passed directly to the function.
        if save_name is not None, the files will be saved in a separate folder

        TODO:
            - Add option for func to be a Callable
            - Is func_index needed at all?
        """
        if hasattr(self, func):
            # Save func as string for dictionaries
            # TODO: Output type should be input after args, kwargs
            self.functions.append(tuple([func, output_type,
                                         args, kwargs, save_name]))
        else:
            raise AttributeError(f"Function {func} not found in {self}.")

        self.func_index = {i: f for i, f in enumerate(self.functions)}

    def run_operation(self,
                      inputs: ImageContainer
                      ) -> ImageContainer:
        """
        Rules for operation functions:
            Must take in at least one of image, mask, track
            Can take in as many of each, but must be a separate positional argument
            Either name or type hint must match the types above
            If multiple, must be present in above order

        TODO:
            - Update rules above
        """
        # By default, return all inputs of output type
        return_container = ImageContainer()
        return_container.update({k: v for k, v in inputs.items()
                                 if k[1] == self._output_type.__name__})

        # fidx is function index
        for fidx, (func, expec_type, args, kwargs, save_name) in enumerate(self.functions):
            func = getattr(self, func)
            last_func = fidx + 1 == len(self.functions)

            # Set up the function run
            self.logger.info(f'Starting function {func.__name__}')
            self.logger.info('All Inputs: '
                             f'{[(key, inpt.shape, inpt.dtype) for key, inpt in inputs.items()]}')
            self.logger.info(f'args / kwargs: {args} {kwargs}')
            self.logger.info(f'User set output type: {expec_type}. Save name: {save_name}')

            # Check if input is already loaded
            if not self.force_rerun:
                if expec_type is None:
                    _out_type = self._get_func_output_type(func)
                else:
                    _out_type = expec_type

                # Only checks for functions that are expected to be saved
                if save_name is not None:
                    _check_key = (save_name, _out_type)
                elif last_func:
                    _check_key = (self.output, _out_type)
                else:
                    _check_key = (None, None)

                # The output already exists
                # TODO: Add ability to use keys with _split_key
                if _check_key in inputs:
                    self.logger.info(f'Output already loaded: {_check_key} '
                                     f'{inputs[_check_key].shape}, '
                                     f'{inputs[_check_key].dtype}.')
                    self.logger.info(f'Skipping {func.__name__}.')
                    continue

            # Get outputs and overwrite type if needed
            exec_timer = time.time()
            output_key, result = func(inputs, *args, **kwargs)
            output_key = [out if expec_type is None else (out[0], expec_type)
                          for out in output_key]
            self.logger.info(f'Returned: {[o for o in output_key]}, '
                             f'{[(r.shape, r.dtype) for r in result]}')

            # Outputs written to inputs ImageContainer before keys are changed
            # for out, res in zip(output_key, result):
            #     inputs[out] = res

            # Update the keys before saving images
            save_folders = []
            if len(result) > 1:
                # By default, use names of the results to identify outputs
                # Remove self._split_key from keys
                res_id = [o[0].split(self._split_key)[0] for o in output_key]

                # Overwrite the output_keys to ensure they are unique
                if len(set([o[1] for o in output_key])) == len(output_key):
                    # Types of outputs are unique
                    res_id = [o[1] for o in output_key]
                elif len(set([o[1] for o in output_key])) != len(output_key):
                    # Neither names or types are unique
                    self.logger.warn(f'Keys are not unique for {func.__name__}'
                                     '. Some outputs may be overwritten.')

                if save_name is not None:
                    # Will save in sub-directories in folder save_name
                    output_key = [(f'{save_name}{self._split_key}{r}', out[1])
                                  for out, r in zip(output_key, res_id)]
                    save_folders = [os.path.join(save_name, r) for r in res_id]
                elif last_func:
                    output_key = [(f'{self.output}{self._split_key}{r}', out[1])
                                  for out, r in zip(output_key, res_id)]
                    save_folders = [os.path.join(save_name, r) for r in res_id]
            else:
                # Only one result, nothing fancy with the outputs
                if save_name is not None:
                    output_key = [(save_name, output_key[0][1])]
                    save_folders = [save_name]
                elif last_func:
                    output_key = [(self.output, output_key[0][1])]
                    save_folders = [self.output]

            # Check if any images need to be saved
            for ridx, (out, res) in enumerate(zip(output_key, result)):
                # Save all outputs
                inputs[out] = res

                if save_folders:
                    # Save images if save_name is not None
                    self.logger.info(f'Adding to save container: '
                                     f'{save_folders[ridx]} '
                                     f'{out[1]}, {res.shape}')
                    self.save_arrays[save_folders[ridx]] = out[1], res

                    self.logger.info(f'Returning to the Pipeline: '
                                     f'{out}, {res.shape}')
                    return_container[out] = res

            self.logger.info(f'{func.__name__} execution time: '
                             f'{time.time() - exec_timer}')

        self.logger.info('Returning to pipeline: '
                         f'{[(k, v.shape) for k, v in return_container.items()]}')
        yield from return_container.items()

    def set_logger(self, logger: logging.Logger) -> None:
        """
        """
        # logger is either a Pipeline or Operation logger
        log_name = logger.name

        # This logs to same file, but records the Operation name
        self.logger = logging.getLogger(f'{log_name}.{self.__name__}')

    def get_inputs_and_outputs(self) -> List[List[tuple]]:
        """
        Returns all inputs and outputs to Pipeline._input_output_handler

        # TODO:
            - This must also change when order of inputs to add_func changes
        """
        # Get all keys for the functions in Operation with save_name
        # f = (func, output_type, args, kwargs, save_name)
        f_keys = [(f[-1], f[1]) if f[1] is not None
                  else (f[-1], self._get_func_output_type(f[0]))
                  for f in self.functions]

        # f_keys for inputs should not be included if no function follows
        f_keys_for_inputs = [f for f in f_keys[:-1] if f[0] is not None]
        f_keys = [f for f in f_keys if f[0] is not None]

        # Get all the inputs that were passed to __init__
        inputs = []
        for i in INPT_NAMES:
            inputs.extend([(g, i) for g in getattr(self, f'input_{i}s')])

        inputs += f_keys_for_inputs + [self.output_id]

        # TODO: This is also possibly inaccurate because save_name overwrites output
        outputs = f_keys + [self.output_id]

        return inputs, outputs

    def _get_func_output_type(self, func: (Callable, str)) -> str:
        """Returns the annotated output type of the function"""
        if isinstance(func, str):
            func = getattr(self, func)

        return inspect.signature(func).return_annotation.__name__

    def _operation_to_dict(self, op_slots: Collection[str] = None) -> Dict:
        """
        Returns a dictionary that fully defines the operation
        """
        # Get attributes to lookup
        base_slots = ['__name__', '__module__', 'save', 'output',
                      'force_rerun', '_output_id']
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
                 input_masks = [],
                 output: str = 'process',
                 save: bool = False,
                 force_rerun: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, force_rerun, _output_id)

        # TODO: For every operation, these should be set in BaseOperation
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
                 force_rerun: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, force_rerun, _output_id)

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
                 force_rerun: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, force_rerun, _output_id)

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
                 force_rerun: bool = True,
                 _output_id: Tuple[str] = None
                 ) -> None:
        """
        channels and regions should be the names that will get saved in the final df
        with the images and masks they correspond to.
        """

        super().__init__(output, save, force_rerun, _output_id)

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
        self.functions = [tuple(['extract_data_from_image', None, [], kwargs, None])]
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
                      inputs: ImageContainer
                      ) -> ImageContainer:
        """
        Add more detailed logging information
        """
        # Get inputs that were saved during __init__
        _, expec_type, args, kwargs, name = self.functions[0]

        # Extract needs separate logging here
        self.logger.info(f"Channels: {list(zip(kwargs['channels'], self.input_images))}")
        self.logger.info(f"Regions: {list(zip(kwargs['regions'], self.input_tracks + self.input_masks))}")
        self.logger.info(f"Metrics: {self._metrics}")
        self.logger.info(f"Added metrics: {list(self._props_to_add.keys())}")
        self.logger.info(f"Condition: {kwargs['condition']}")

        return super().run_operation(inputs)


class BaseEvaluate(Operation):
    __name__ = 'Evaluate'
    _input_type = (Arr,)
    _output_type = Arr

    def __init__(self,
                 input_arrays: Collection[str] = [],
                 output: str = 'evaluate',
                 save: bool = False,
                 force_rerun: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, force_rerun, _output_id)

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
