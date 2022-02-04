import os
import warnings
import logging
import time
import inspect
from typing import Collection, Tuple, Callable, List, Dict

import numpy as np
import skimage.measure as meas

from cellst.core.arrays import ConditionArray
from cellst.utils._types import (Image, Mask, Track, Arr, Same,
                                 ImageContainer, INPT_NAMES,
                                 RandomNameProperty)
from cellst.utils.operation_utils import (track_to_mask, parents_from_track,
                                          match_labels_linear)
from cellst.utils.log_utils import get_console_logger
from cellst.utils.utils import ImageHelper
import cellst.utils.metric_utils as metric_utils
import cellst.utils.filter_utils as filter_utils


class Operation():
    __name__ = 'Operation'

    def __init__(self,
                 images: Collection[str] = [],
                 masks: Collection[str] = [],
                 tracks: Collection[str] = [],
                 arrays: Collection[str] = [],
                 output: str = 'out',
                 save: bool = True,
                 force_rerun: bool = False,
                 _output_id: Tuple[str] = None,
                 _split_key: str = '&',
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

        # These inputs will be [] if not used
        if isinstance(images, str):
            self.images = [images]
        else:
            self.images = images

        if isinstance(masks, str):
            self.masks = [masks]
        else:
            self.masks = masks

        if isinstance(tracks, str):
            self.tracks = [tracks]
        else:
            self.tracks = tracks

        if isinstance(arrays, str):
            self.arrays = [arrays]
        else:
            self.arrays = arrays

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
        inputs = tuple([f"{name[0]}:{getattr(self, f'{name}s')}"
                        for name in INPT_NAMES
                        if hasattr(self, f'{name}s')])

        # Format each function as a str
        funcs = [self._format_function_string(func, kwargs)
                 for (func, _, _, kwargs) in self.functions]
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
            if getattr(self, f'{name}s'):
                self.logger.info(f"{name}:{getattr(self, f'{name}s')}")
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
                                  func: str, *,
                                  save_as: str = None,
                                  output_type: str = None,
                                  **kwargs
                                  ) -> None:
        """
        TODO:
            - Add option for func to be a Callable
        """
        if hasattr(self, func):
            # Save func as string for dictionaries
            self.functions.append(tuple([func, save_as, output_type, kwargs]))
        else:
            raise AttributeError(f"Function {func} not found in {self}.")

    def run_operation(self,
                      inputs: ImageContainer
                      ) -> ImageContainer:
        """
        """
        # By default, return all inputs of output type
        return_container = ImageContainer()
        return_container.update({k: v for k, v in inputs.items()
                                 if k[1] == self._output_type.__name__})

        # fidx is function index
        for fidx, (func, save_as, user_type, kwargs) in enumerate(self.functions):
            func = getattr(self, func)
            out_type = user_type if user_type else self._get_func_output_type(func)
            last_func = fidx + 1 == len(self.functions)

            # Set up the function
            self.logger.info(f'Starting function {func.__name__}')
            self.logger.info(
                'All Inputs: '
                f'{[(key, inpt.shape, inpt.dtype) for key, inpt in inputs.items()]}'
            )
            self.logger.info(f'kwargs: {kwargs}')
            self.logger.info(f'output type: {out_type}. '
                             f'save as: {save_as}')

            # Check if input is already loaded
            if not self.force_rerun:
                # Only checks for functions that are expected to be saved
                if save_as:
                    check_key = (save_as, out_type)
                elif last_func:
                    check_key = (self.output, out_type)
                else:
                    check_key = (None, None)

                # The output already exists - skip the function
                # TODO: Add ability to use keys with _split_key
                if check_key in inputs:
                    self.logger.info(f'Output already loaded: {check_key} '
                                     f'{inputs[check_key].shape}, '
                                     f'{inputs[check_key].dtype}.')
                    self.logger.info(f'Skipping {func.__name__}.')
                    continue

            # Run function, get outputs, and overwrite type if needed
            exec_timer = time.time()
            output_key, result = func(inputs, **kwargs)
            output_key = [out if not user_type else (out[0], user_type)
                          for out in output_key]
            self.logger.info(f'Returned: {[o for o in output_key]}, '
                             f'{[(r.shape, r.dtype) for r in result]}')

            # Change keys for saving if needed
            ret_to_pipe = save_as or last_func
            save_folders = []
            if len(result) > 1:
                # Use names of the results to identify outputs
                # Remove self._split_key from keys
                names = [o[0].split(self._split_key)[0] for o in output_key]
                types = [o[1] for o in output_key]

                # Overwrite the output_keys to ensure they are unique
                if len(set(names)) == len(output_key):
                    res_id = names
                elif len(set(types)) == len(output_key):
                    # Types of outputs are unique
                    res_id = types
                else:
                    # Neither names or types are unique
                    res_id = names
                    self.logger.warn(f'Keys are not unique for {func.__name__}'
                                     '. Some outputs may be overwritten.')

                # Create new output_keys if needed
                if save_as:
                    # Will save in sub-directories in folder save_as
                    output_key = [(f'{save_as}{self._split_key}{r}', out[1])
                                  for out, r in zip(output_key, res_id)]
                    save_folders = [os.path.join(save_as, r) for r in res_id]
                elif last_func:
                    output_key = [(f'{self.output}{self._split_key}{r}', out[1])
                                  for out, r in zip(output_key, res_id)]
                    if self.save:
                        save_folders = [os.path.join(self.output, r)
                                        for r in res_id]
            else:
                # Only one result, nothing fancy with the outputs
                if save_as:
                    output_key = [(save_as, output_key[0][1])]
                    save_folders = [save_as]
                elif last_func:
                    output_key = [(self.output, output_key[0][1])]
                    if self.save:
                        save_folders = [self.output]

            # Check if any images need to be saved
            for ridx, (out, res) in enumerate(zip(output_key, result)):
                # Put outputs in input container for future functions
                inputs[out] = res

                if ret_to_pipe:
                    self.logger.info(f'Returning to the Pipeline: '
                                     f'{out}, {res.shape}, {res.dtype}')
                    return_container[out] = res

                if save_folders:
                    # Save images if save_as is not None
                    self.logger.info(f'Adding to save container: '
                                     f'{save_folders[ridx]} '
                                     f'{out[1]}, {res.shape}')
                    self.save_arrays[save_folders[ridx]] = out[1], res

            self.logger.info(f'{func.__name__} execution time: '
                             f'{time.time() - exec_timer}')

        self.logger.info(
            'Returning to pipeline: '
            f'{[(k, v.shape, v.dtype) for k, v in return_container.items()]}'
        )
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
        Returns all possible inputs and outputs expected.
        It's likely not all will actually get made/used.

        Needed for Pipeline._input_output_handler

        TODO: This whole system should be handled differently.
              Inputs and outputs are now basically the same, so
              one of them should be removed.
        """
        # Get all keys for the functions in Operation with save_as
        f_keys = [(svname, usr_typ) if usr_typ
                  else (svname, self._get_func_output_type(fnc))
                  for (fnc, svname, usr_typ, _) in self.functions]
        f_keys_for_inputs = [f for f in f_keys if f[0]]
        f_keys = [f for f in f_keys if f[0]]

        # Get all the inputs that were passed to __init__
        inputs = []
        for i in INPT_NAMES:
            inputs.extend([(g, i) for g in getattr(self, f'{i}s')])

        # Check if save_as was set for last function
        if self.functions[-1][1]:
            last_name = self.functions[-1][1]
        else:
            last_name = self.output_id
        last_key = (last_name, self.output_id[1])
        outputs = f_keys + [last_key]

        # Do not include function outputs in inputs if force_rerun
        if not self.force_rerun:
            inputs += f_keys_for_inputs + [last_key]

        return inputs, outputs

    def _get_func_output_type(self, func: (Callable, str)) -> str:
        """Returns the annotated output type of the function"""
        if isinstance(func, str):
            func = getattr(self, func)

        return inspect.signature(func).return_annotation.__name__

    def _operation_to_dict(self) -> Dict:
        """
        Returns a dictionary that fully defines the operation
        """
        # Get attributes to lookup and save in dictionary
        base_slots = ('__name__', '__module__', 'images', 'masks',
                      'tracks', 'arrays', 'save', 'output',
                      'force_rerun', '_output_id', '_split_key')
        op_defs = {}
        for att in base_slots:
            op_defs[att] = getattr(self, att, None)

        # Save function definitions
        func_defs = {}

        for (func, save_as, user_type, kwargs) in self.functions:
            # Rename if already in dictionary
            count = 1
            key = func
            while key in func_defs:
                key = func + f'_{count}'
                count += 1

            func_defs[key] = {}
            func_defs[key]['func'] = func
            try:
                func_defs[key]['output_type'] = user_type.__name__
            except AttributeError:
                func_defs[key]['output_type'] = user_type
            func_defs[key]['name'] = save_as
            func_defs[key]['kwargs'] = kwargs

        # Save in original dictionary
        op_defs['_functions'] = func_defs

        return op_defs

    def _format_function_string(self,
                                fname: str,
                                kwargs: dict
                                ) -> str:
        """
        Nicely formats the function specifications for the Operation

        TODO: is there a way to neatly include type and save name?
        """
        # Format kwargs to str
        if len(kwargs) > 0:
            str_kwargs = ', '.join(tuple([f'{k}={v}'
                                          for k, v in kwargs.items()]))
        else:
            str_kwargs = ''
        return f'{fname}({str_kwargs})'

    ### Adding very basic functions to be inherited ###
    @ImageHelper(by_frame=False, as_tuple=True)
    def image_to_mask(self,
                       image: Image,
                       ) -> Mask:
        """
        TODO: Should be able to wrtie a complete function
              to handle all types, but that might require
              some work in ImageHelper
        """
        return image

    @ImageHelper(by_frame=False, as_tuple=True)
    def image_to_track(self,
                       image: Image,
                       ) -> Track:
        return image

    @ImageHelper(by_frame=False, as_tuple=True)
    def mask_to_image(self,
                      mask: Mask,
                      ) -> Image:
        return mask

    @ImageHelper(by_frame=False, as_tuple=True)
    def mask_to_track(self,
                      mask: Mask,
                      ) -> Track:
        return mask

    @ImageHelper(by_frame=False, as_tuple=True)
    def track_to_mask(self,
                      track: Track,
                      ) -> Mask:
        return track

    @ImageHelper(by_frame=False, as_tuple=True)
    def track_to_image(self,
                       track: Track,
                       ) -> Image:
        return track

    @ImageHelper(by_frame=True)
    def apply_mask(self,
                   image: Image,
                   mask: Mask = None,
                   mask_name: str = None,
                   *args, **kwargs
                   ) -> Same:
        """
        Applies a boolean mask.
        if mask_name is given, overrides mask

        mask_name can be any mask in filter_utils
        *args and **kwargs passed there.
        """
        if mask_name:
            mask = getattr(filter_utils, mask_name)
            mask = mask(image, *args, **kwargs).astype(bool)
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(bool)
        else:
            warnings.warn('Did not get useable mask.', UserWarning)
            mask = np.ones(image.shape, dtype=bool)

        return np.where(mask, image, 0)

    @ImageHelper(by_frame=True)
    def match_labels_linear(self,
                            dest: Mask,
                            source: Mask
                            ) -> Mask:
        return match_labels_linear(source, dest)


class BaseProcessor(Operation):
    __name__ = 'Processor'
    _input_type = (Image,)
    _output_type = Image

    def __init__(self,
                 output: str = 'process',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)

    def __call__(self,
                 images: Collection[Image] = [],
                 ) -> Image:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, [], [], [])


class BaseSegmenter(Operation):
    __name__ = 'Segmenter'
    _input_type = (Image,)
    _output_type = Mask

    def __init__(self,
                 output: str = 'mask',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)

    def __call__(self,
                 images: Collection[Image] = [],
                 masks: Collection[Mask] = []
                 ) -> Mask:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, masks, [], [])


class BaseTracker(Operation):
    __name__ = 'Tracker'
    _input_type = (Image, Mask)
    _output_type = Track

    def __init__(self,
                 output: str = 'track',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)

    def __call__(self,
                 images: Collection[Image] = [],
                 masks: Collection[Mask] = []
                 ) -> Track:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation(images, masks, [], [])


class BaseExtractor(Operation):
    """TODO: Include add_derived_metrics"""
    __name__ = 'Extractor'
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
    _derived_metrics = {}
    _filters = {}

    # _props_to_add is what actually gets used to decide on extra metrics
    _props_to_add = {}

    def __init__(self,
                 images: Collection[str] = [],
                 masks: Collection[str] = [],
                 tracks: Collection[str] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 lineages: Collection[np.ndarray] = [],
                 condition: str = 'condition',
                 position_id: int = 0,
                 min_trace_length: int = 0,
                 remove_parent: bool = True,
                 output: str = 'data_frame',
                 save: bool = True,
                 force_rerun: bool = True,
                 _output_id: Tuple[str] = None,
                 **kwargs
                 ) -> None:
        """
        """
        super().__init__(images, masks, tracks,
                         output=output, save=save,
                         force_rerun=force_rerun,
                         _output_id=_output_id, **kwargs)

        if isinstance(regions, str):
            regions = [regions]
        if isinstance(channels, str):
            channels = [channels]

        if not channels:
            channels = self.images

        # Prefer tracks for naming
        if not regions:
            if tracks:
                regions = tracks
            else:
                regions = masks

        # These kwargs get passed to self.extract_data_from_image
        kwargs = dict(channels=channels, regions=regions, lineages=lineages,
                      condition=condition, min_trace_length=min_trace_length,
                      remove_parent=remove_parent, position_id=position_id)
        # Add extract_data_from_image
        # functions are expected to be (func, save_as, user_type, kwargs)
        self.functions = [tuple(['extract_data_from_image', output, None, kwargs])]

        # Add division_frame and parent_id
        # TODO: Make optional
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
        return (self._metrics
               + self._extra_properties
               + list(self._props_to_add.keys()))

    @property
    def extract_kwargs(self) -> dict:
        return self.functions[0][-1]  # index for kwargs passed to extract_data

    def _correct_metric_dim(self, met_list: List[str]) -> List[str]:
        """
        Adjust all measures for the presence of multi-scalar metrics
        """
        new_met_list = []
        for m in met_list:
            if m == 'bbox':
                # becomes bbox-1...bbox-4:(min_row, min_col, max_row, max_col)
                new_met_list.extend(['min_y', 'min_x', 'max_y', 'max_x'])
            elif m == 'centroid':
                # becomes centroid-1, centroid-2:(y, x)
                new_met_list.extend(['y', 'x'])
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
        self._metric_idx = {k: i for i, k in enumerate(all_metrics)}

        # Build output
        out = np.empty((image.shape[0], len(all_metrics), len(cell_index)))
        # TODO: Add option to use different pad
        out[:] = np.nan

        for frame in range(image.shape[0]):
            # Extract metrics from each region in frame
            rp = meas.regionprops_table(mask[frame], image[frame],
                                        properties=metrics,
                                        extra_properties=extra_metrics)
            # frame_data.shape is (len(metrics), len(cells))
            frame_data = np.row_stack(tuple(rp.values()))

            # Label is in the first position
            for n, lab in enumerate(frame_data[0, :]):
                # Cast to int for indexing
                lab = int(lab)

                if lab in daughter_to_parent:
                    # TODO: This should be optional (or at least the
                    #       addition of division_frame and parent_id)
                    # Get parent label
                    # NOTE: Could this ever raise a KeyError?
                    par = daughter_to_parent[lab]

                    # Copy parent trace to location of daughter trace
                    # Everything after frame is overwritten by daughter trace
                    out[:, :, cell_index[lab]] = out[:, :, cell_index[par]]

                    # Add division data to frame_data before saving
                    try:
                        frame_data[self._metric_idx['division_frame'], n] = frame
                    except KeyError:
                        pass

                    try:
                        frame_data[self._metric_idx['parent_id'], n] = par
                    except KeyError:
                        pass

                # Save frame data
                out[frame, :, cell_index[lab]] = frame_data[:, n]

        return np.moveaxis(out, 0, -1)

    def _operation_to_dict(self) -> Dict:
        op_dict = super()._operation_to_dict()

        # Add the kwargs for extract_data_from_image
        op_dict.update(self.extract_kwargs)

        # Add metrics and extra properties
        # TODO: This is also a bit hackish
        func = 'extract_data_from_image'
        op_dict['_functions'][func]['metrics'] = self._metrics
        op_dict['_functions'][func]['derived_metrics'] = self._derived_metrics
        op_dict['_functions'][func]['extra_props'] = self._props_to_add

        return op_dict

    def _calculate_derived_metrics(self, array: ConditionArray) -> None:
        """
        Does the actual computations from add_derived_metric
        """
        for name, (func, keys, args, kwargs) in self._derived_metrics.items():
            self.logger.info(f'Calculating derived metric {name}')
            # NOTE: Only two arrays for now
            arrs = [array[tuple(k)] for k in keys]
            func = getattr(np, func)

            # Assume the function takes two arrays for now
            result = func(arrs[0], arrs[1], *args, **kwargs)
            inv_result = func(arrs[1], arrs[0], *args, **kwargs)

            # Get channel and region from keys for saving
            save_key = [name] + [k for k in keys[0]
                                 if k not in self._metric_idx]
            inv_save_key = [name] + [k for k in keys[1]
                                     if k not in self._metric_idx]

            # Metric slots should already by in Condition array
            array[tuple(save_key)] = result
            array[tuple(inv_save_key)] = inv_result

    def _apply_filters(self, array: ConditionArray) -> None:
        """Removes cells based on user-defined filters"""
        for name, (metr, reg, chn, args, kws) in self._filters.items():
            self.logger.info(f'Removing cells with filter {name}')
            self.logger.info(f'Inputs: {[reg, chn, metr, args, kws]}')
            self.logger.info(f'Current array size: {array.shape}')
            mask = array.generate_mask(name, metr, reg, chn,
                                       key=None, *args, **kws)
            array.filter_cells(mask, delete=True)
            self.logger.info(f'Post-filter array size: {array.shape}')

    def add_extra_metric(self, name: str, func: Callable = None) -> None:
        """
        Allows for adding custom metrics. If function is none, value will just
        be nan.

        TODO: Callable function won't be saveable in YAML files
        """
        if not func:
            if name in self._possible_metrics:
                self._metrics.append(name)
            else:
                try:
                    func = getattr(metric_utils, name)
                except AttributeError:
                    # Function not implemented by me
                    func = RandomNameProperty()
        else:
            assert isinstance(func, Callable)

        self._props_to_add[name] = func

    def add_derived_metric(self,
                           metric_name: str,
                           keys: Collection[Tuple[str]],
                           func: str = 'sum',
                           *args, **kwargs
                           ) -> None:
        """
        Calculates additional metrics based on information already in array
        func can be any numpy function - expected to pass 2 arrays though

        TODO: Add ability to have more than 2 arrays
        TODO: Add ability to propagate results to other keys
        TODO: Add keys to save this in yaml dictionary
        TODO: Add possiblity for custom Callable function
        """
        # Check the inputs now before calculation
        # Only two inputs allowed for now
        assert len(keys) == 2
        # Assert that keys include channel, region, and metric
        for key in keys:
            assert len(key) == 3
        assert hasattr(np, func), 'Derived metric must be numpy function'

        # Save to calculated metrics to get added after extract is done
        self._derived_metrics[metric_name] = tuple([func, keys,
                                                    args, kwargs])

        # Fill in the metric with just nan for now
        self._props_to_add[metric_name] = RandomNameProperty()

        self.logger.info(f'Added derived metric {metric_name}')

    def add_filter(self,
                   filter_name: str,
                   metric: str,
                   region: [str, int] = 0,
                   channel: [str, int] = 0,
                   *args, **kwargs
                   ) -> None:
        """
        TODO: Add ability to pass Callable, has to be done after Extract now
        """
        assert hasattr(filter_utils, filter_name), f'{filter_name} not found.'
        added_metrics = (self._extra_properties
                         + self._metrics
                         + list(self._derived_metrics.keys()))
        assert metric in added_metrics, f'Metric {metric} not found'

        self._filters[filter_name] = tuple([metric, region, channel,
                                            args, kwargs])

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
        if not_allowed:
            warnings.warn(f'Metrics {[not_allowed]} are not supported. Use '
                          'CellArray.add_extra_metric to add custom metrics.')

    def add_function_to_operation(self,
                                  func: str, *,
                                  output_type: str = None,
                                  name: str = None,
                                  **kwargs
                                  ) -> None:
        """
        Extractor currently only supports one function due to how
        extract_data_from_images expects the inputs.
        """
        raise NotImplementedError('Adding new functions to Extractor is '
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
        kwargs = self.functions[0][-1]

        # Log inputs to Extractor
        self.logger.info(f"Channels: {list(zip(kwargs['channels'], self.images))}")
        self.logger.info(f"Regions: {list(zip(kwargs['regions'], self.tracks + self.masks))}")
        self.logger.info(f"Metrics: {self._metrics}")
        self.logger.info(f"Added metrics: {list(self._props_to_add.keys())}")
        self.logger.info(f"Condition: {kwargs['condition']}")

        return super().run_operation(inputs)


class BaseEvaluator(Operation):
    __name__ = 'Evaluator'
    _input_type = (Arr,)
    _output_type = Arr

    def __init__(self,
                 output: str = 'evaluate',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)

    def __call__(self,
                 arrs: Collection[Arr]
                 ) -> Arr:
        """
        Calls run_operation. This is intended to be
        used independently of Pipeline.
        """
        return self.run_operation([], [], [], arrs)
