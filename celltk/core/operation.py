import os
import warnings
import logging
import time
import inspect
import itertools
from copy import deepcopy
from typing import (Collection, Tuple, Callable,
                    List, Dict, Generator, Union)

import numpy as np
import skimage.measure as meas
import skimage.morphology as morph
import skimage.util as util
import SimpleITK as sitk

from celltk.core.arrays import ConditionArray
from celltk.utils._types import (Image, Mask, Array, Stack,
                                 ImageContainer, INPT_NAMES,
                                 RandomNameProperty)
from celltk.utils.operation_utils import (track_to_mask, parents_from_track,
                                          match_labels_linear, mask_to_seeds)
from celltk.utils.log_utils import get_console_logger
from celltk.utils.utils import ImageHelper
import celltk.utils.metric_utils as metric_utils
import celltk.utils.filter_utils as filter_utils


class Operation:
    """
    Base class for all other Operations.

    :param images: Names of images to use in this operation
    :param masks: Names of masks to use in this operation
    :param arrays: Names of arrays to use in this operation
    :param output: Name to save the output stack
    :param save: If False, the final result will not be saved
        to disk.
    :param force_rerun: If True, all functions are run even
        if the output stack is already loaded.
    :param _output_id: Private attribute used to track output
        from the Operation.
    :param _split_key: Used to specify outputs from functions that
        return multiple outputs. For example, if you align two channels
        the outputs will be saved as 'align&channel000' and
        'align&channel001' for _split_key = '&'
    """
    __name__ = 'Operation'

    def __init__(self,
                 images: Collection[str] = [],
                 masks: Collection[str] = [],
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
                 arrays: Collection[Array] = [],
                 _return_keys: bool = False
                 ) -> Union[Image, Mask, Array]:
        """
        __call__ runs operation independently of Pipeline class
        """
        # Cast all to lists if they are not
        if not isinstance(images, (tuple, list)): images = [images]
        if not isinstance(masks, (tuple, list)): masks = [masks]
        if not isinstance(arrays, (tuple, list)): arrays = [arrays]

        # Generate keys based on enumeration
        container = ImageContainer()
        inputs = [Image, Mask, Array]
        for nm, stack in zip(inputs, [images, masks, arrays]):
            if stack:
                nm = nm.__name__
                for i, st in enumerate(stack):
                    key = (f'{nm}_{i}', nm)
                    container[key] = st

        out = self.run_operation(container, _return_inputs=False)
        if _return_keys:
            return tuple(dict(out).items())
        else:
            return list(dict(out).values())

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
            if hasattr(self, f'{name}s'):
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

    def add_function(self,
                     func: str, *,
                     save_as: str = None,
                     output_type: str = None,
                     **kwargs
                     ) -> None:
        """Adds a function to be run by the Operation

        :param func: Name of the function. Must exist in
            the Operation class.
        :param save_as: Name to save the result as. If not
            set, will not be saved unless it is the last
            function and Operation.save is True.
        :param output_type: If given, changes the default output type to
            another CellTK type (i.e iimage, mask, track)
        :pararm kwargs: All other kwargs are passed to the function.
            Note that they must be provided as kwargs.

        :return: None

        TODO:
            - Add option for func to be a Callable
        """
        if hasattr(self, func):
            # Save func as string for dictionaries
            self.functions.append(tuple([func, save_as, output_type, kwargs]))
        else:
            raise AttributeError(f"Function {func} not found in {self}.")

    def run_operation(self,
                      inputs: ImageContainer,
                      _return_inputs: bool = True
                      ) -> Generator:
        """Sets up a generator to run the operation and return
        results for each function in turn. This function also
        handles naming and typing the outputs of each function.
        Typically used by ``Pipeline``. To use an ``Operation`` on
        arrays and get arrays in return, preferably use the
        __call__ method instead.

        :param inputs: ``ImageContainer`` with all of the inputs
            required by the ``Operation``.

        :return: A signle generator for the results of each function.
        """
        # By default, return all inputs of output type
        return_container = ImageContainer()
        if _return_inputs:
            return_container.update({k: v for k, v in inputs.items()
                                     if k[1] == self._output_type.__name__})

        if not self.functions:
            warnings.warn(f'No functions found in {self}.', UserWarning)
            yield from return_container.items()

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

                # If outputs are already loaded, skip the function
                if self._check_if_loaded(check_key, inputs):
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
        """Add a custom logger object to log Pipeline operation.

        :param logger: Custom logging object. Must be type logging.Logger.

        :return: None
        """
        # logger is either a Pipeline or Operation logger
        log_name = logger.name

        # This logs to same file, but records the Operation name
        self.logger = logging.getLogger(f'{log_name}.{self.__name__}')

    def get_inputs_and_outputs(self) -> List[List[tuple]]:
        """Returns all possible inputs and outputs expected
        by the Operation. Not all will be made or used. Typically
        this function is used by Pipeline to determine which files
        to load.

        NOTE:
            - This function is expected to change soon in a new
              version.

        TODO:
            - This whole system should be handled differently.
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
            # Messy fix, but not folder for Stack/Stack input
            if i != Stack.__name__:
                inputs.extend([(g, i) for g in getattr(self, f'{i}s')])

        # Check if save_as was set for last function
        if self.functions and self.functions[-1][1]:
            # TODO: If save name was set, this is already in outputs
            last_name = (self.functions[-1][1],
                         self.output_id[1])
        else:
            last_name = self.output_id

        outputs = f_keys + [last_name]

        # Do not include function outputs in inputs if force_rerun
        if not self.force_rerun:
            inputs += f_keys_for_inputs + [last_name]

        return inputs, outputs

    def _check_if_loaded(self,
                         check_key: Tuple[str],
                         inputs: ImageContainer
                         ) -> bool:
        """Checks if the expected output of a function has
        already been loaded.

        :param check_key: Name and type of the expected output.
        :param inputs: ImageContainer with the loaded images.

        :return: True if the image has been loaded. False otherwise.
        """

        if check_key not in inputs:
            # Try with the split_key
            _split = lambda x: x[0].split(self._split_key)[0]
            matches = [k for k in inputs.keys()
                       if _split(k) == check_key[0]]
            if not matches:
                return False
            else:
                # TODO: Matches now need to be filtered by type
                pass
        else:
            matches = [check_key]

        # Log information if something is found
        self.logger.info(f'Output for {check_key} already loaded.')
        for m in matches:
            self.logger.info(f'{m}, {inputs[m].shape}, {inputs[m].dtype}.')

        return True

    def _get_func_output_type(self, func: (Callable, str)) -> str:
        """Returns the annotated output type of the given function.

        :param func: Function to inspect.

        :return: Name of expected output type
        """
        if isinstance(func, str):
            func = getattr(self, func)

        return inspect.signature(func).return_annotation.__name__

    def _operation_to_dict(self) -> Dict[str, str]:
        """Returns a dictionary that fully defines the Operation and
        all of the user supplied parameters.

        :return: Dictionary defining the Operation and parameters.
        """
        # Get attributes to lookup and save in dictionary
        base_slots = ('__name__', '__module__', 'images', 'masks',
                      'arrays', 'save', 'output',
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
        """Formats the function specifications for the function to
        be displayed to user or logged.

        :param fname: Name of the function
        :param kwargs: User supplied kwargs to the function.

        TODO:
            - Is there a way to neatly include type and save name?
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
        """Convert an Image stack to a Mask stack.

        :param image: Image stack to be typed as a Mask

        :return: Input stack with Mask designation.

        TODO:
            - Should be able to wrtie a complete function
              to handle all types, but that might require
              some work in ImageHelper
        """
        return image

    @ImageHelper(by_frame=False, as_tuple=True)
    def mask_to_image(self,
                      mask: Mask,
                      ) -> Image:
        """Convert a Mask stack to a Image stack.

        :param mask: Mask stack to be typed as a Image

        :return: Input stack with Image designation.
        """
        return mask

    @ImageHelper(by_frame=True)
    def apply_mask(self,
                   image: Image,
                   mask: Mask = None,
                   mask_name: str = None,
                   *args, **kwargs
                   ) -> Stack:
        """ Applies a boolean mask to an image. User can supply
        a boolean mask or the name of a function from filter_utils.

        :param image: Image to be masked
        :param mask: Boolean mask of same shape as image
        :param mask_name: Name of a function in filter_utils to
            use to create a boolean mask. If given, any input
            mask is ignored
        :param args: Passed to mask_name function
        :param kwargs: Passed to mask_name function.

        :return: Masked image
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
    def make_boolean_mask(self,
                          image: Image,
                          mask_name: str = 'outside',
                          *args, **kwargs
                          ) -> Mask:
        """Generates a mask using a function from filter_utils.

        :param image: Image to use for generating mask
        :param mask_name: Name of the function in filter_utils
        :param args: Passed to mask_name function
        :param kwargs: Passed to mask_name function.

        :return: Boolean mask with same shape as input image
        """
        mask = getattr(filter_utils, mask_name)
        return mask(image, *args, **kwargs).astype(bool)

    @ImageHelper(by_frame=True)
    def match_labels_linear(self,
                            dest: Mask,
                            source: Mask
                            ) -> Mask:
        """Transfers labels from source mask to dest mask based on
        maximizing the area overlap. Objects with no overlap are given
        a new label starting at max_label_value + 1.

        :param dest: Mask with objects to be labeled
        :param source: Mask with labeled objects to use as template

        :return: Labeled mask
        """
        return match_labels_linear(source, dest)

    @ImageHelper(by_frame=False)
    def invert(self,
               image: Image,
               signed_float: bool = False
               ) -> Image:
        """Inverts the intensity of the given image. Wrapper for
        skimage.util.invert.

        :param image: Image to be inverted.
        :param signed_float: Set to True if _______________"""
        return util.invert(image, signed_float)

    @ImageHelper(by_frame=True)
    def regular_seeds(self,
                      image: Image,
                      n_points: int = 25,
                      dtype: type = np.uint8
                      ) -> Mask:
        """Labels an arbitrary number of approximately evenly-spaced
        pixels with unique integers.

        :param image: Images to serve as template for making seed mask
        :param n_points: Number of pixels to label
        :param dtype: Data type of the output array.

        :return: Mask of same shape as image with n_points pixels labeled
            with a unique integer.
        """
        return util.regular_seeds(image.shape, n_points, dtype)

    @ImageHelper(by_frame=True)
    def mask_to_seeds(self,
                      mask: Mask
                      ) -> Mask:
        """Create seed points at the centroid of each object in mask.

        :param mask: Mask to serve as template

        :return: Mask of same shape as input mask with the pixel corresponding
            to the centroid of each object labeled."""
        return mask_to_seeds(mask)

    @ImageHelper(by_frame=True)
    def binary_threshold(self,
                         image: Image,
                         lower: float = None,
                         upper: float = None,
                         inside: float = None,
                         outside: float = None
                         ) -> Image:
        """
        This function is too much. Write a simpler binarize function
        This is more like constant_thres w/o label as is.
        """
        # Set up the filter
        fil = sitk.BinaryThresholdImageFilter()
        if lower is not None: fil.SetLowerThreshold(lower)
        if upper is not None: fil.SetUpperThreshold(upper)
        if inside is not None: fil.SetInsideValue(inside)
        if outside is not None: fil.SetOutsideValue(outside)

        img = sitk.GetImageFromArray(image)
        return sitk.GetArrayFromImage(fil.Execute(img))


class BaseProcess(Operation):
    """
    Base class for processing operations. Typically used to apply image filters
    such as blurring or edge detection.
    """
    __name__ = 'Process'
    _input_type = (Image,)
    _output_type = Image

    def __init__(self,
                 output: str = 'process',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)


class BaseSegment(Operation):
    """
    Base class for segmenting operations. Typically used to create masks
    from images or to adjust and label already existing masks.
    """
    __name__ = 'Segment'
    _input_type = (Image,)
    _output_type = Mask

    def __init__(self,
                 output: str = 'mask',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)


class BaseTrack(Operation):
    """
    Base class for tracking operations. Typically used to link objects
    in a provided mask. Can also be used to detect dividing cells.
    """
    __name__ = 'Track'
    _input_type = (Image, Mask)
    _output_type = Mask

    def __init__(self,
                 output: str = 'track',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)


class BaseExtract(Operation):
    """
    Base class for extracting operations. Typically used to extract data
    from an intensity image using masks as a guide.

    :param images: Images to extract data from.
    :param masks: Masks to segment images with.
    :param channels: Names of channels corresponding to
    :param regions: Names of segmented regions corresponding,
        tracks and masks, in that order.
    :param lineages: Lineage files corresponding to masks if provided.
    :param time: If int or float, designates time between frames.
        If array, marks the frame time points.
    :param condition: Name of the condition
    :param position_id: Unique identifier if multiple ConditionArrays
        will share the same condition
    :param min_trace_length: All cells with shorter traces will
        be deleted from the final array. Length of trace is determined
        by number of nans in the 'label' metric
    :param remove_parent: If true, parents of cells are not kept in
        the final ConditionArray.
    :param output: Name to save the output stack
    :param save: If False, the final result will not be saved
        to disk.
    :param force_rerun: If True, all functions are run even
        if the output stack is already loaded.
    :param skip_frames: Use to specify frames to be skipped. If provided
        to Pipeline, does not need to be provided again, but must match.
    :param _split_key: Used to specify outputs from functions that
        return multiple outputs. For example, if you align two channels
        the outputs will be saved as 'align&channel000' and
        'align&channel001' for _split_key = '&'
    """
    __name__ = 'Extract'
    _input_type = (Image, Mask)
    _output_type = Array
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

    _metrics = ('label', 'area', 'convex_area', 'filled_area', 'bbox',
                'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                'minor_axis_length', 'major_axis_length',
                'orientation', 'perimeter', 'solidity')
    _extra_properties = ['division_frame', 'parent_id', 'total_intensity',
                         'median_intensity']

    def __init__(self,
                 images: Collection[str] = [],
                 masks: Collection[str] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 lineages: Collection[np.ndarray] = [],
                 time: float = None,
                 condition: str = 'condition',
                 position_id: int = 0,
                 min_trace_length: int = 0,
                 remove_parent: bool = True,
                 output: str = 'data_frame',
                 save: bool = True,
                 force_rerun: bool = True,
                 skip_frames: Tuple[int] = tuple([]),
                 _output_id: Tuple[str] = None,
                 **kwargs
                 ) -> None:
        """
        """
        super().__init__(images, masks,
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
            regions = masks

        # These kwargs get passed to self.extract_data_from_image
        kwargs = dict(images=images, masks=masks,
                      channels=channels, regions=regions, lineages=lineages,
                      condition=condition, min_trace_length=min_trace_length,
                      remove_parent=remove_parent, position_id=position_id,
                      skip_frames=skip_frames, time=time)
        # Add extract_data_from_image
        # functions are expected to be (func, save_as, user_type, kwargs)
        self.functions = [tuple(['extract_data_from_image',
                                 output, None, kwargs])]

        # These cannot be class attributes
        self._derived_metrics = {}
        self._filters = []
        # _props_to_add is what actually gets used to decide on extra metrics
        self._props_to_add = {}

        # Add division_frame and parent_id
        # TODO: Make optional
        for m in self._extra_properties:
            self.add_extra_metric(m)

    def __call__(self,
                 images: Collection[Image],
                 masks: Collection[Mask] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 lineages: Collection[np.ndarray] = [],
                 condition: str = None,
                 **kwargs,
                 ) -> Array:
        """
        This directly calls extract_data_from_image
        instead of using run_operation
        """
        kwargs.update(dict(channels=channels, regions=regions,
                           condition=condition, lineages=lineages))
        return self.extract_data_from_image.__wrapped__(self, images, masks,
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
                                 track: Mask,
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
        frames = image.shape[0] + len(self.skip_frames)
        out = np.empty((frames, len(all_metrics), len(cell_index)))
        # TODO: Add option to use different pad
        out[:] = np.nan

        # adj_frame accounts for skip_frames
        adj_frame = 0
        for _frame in range(frames):
            if _frame in self.skip_frames:
                adj_frame += 1
                continue
            else:
                frame = _frame - adj_frame
                # This should never fail
                assert frame >= 0

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
                out[_frame, :, cell_index[lab]] = frame_data[:, n]

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
        op_dict['_functions'][func]['filters'] = self._filters
        op_dict['_functions'][func]['extra_props'] = self._props_to_add

        return op_dict

    def _calculate_derived_metrics(self, array: ConditionArray) -> None:
        """
        Does the actual computations from add_derived_metric
        """
        for name, (func, keys, incl, prop, frm, args, kwargs) in self._derived_metrics.items():
            self.logger.info(f'Calculating derived metric {name}')
            if frm:
                try:
                    # If only a number - take that many frames from start
                    _rng = slice(None, int(frm))
                except TypeError:
                    _rng = slice(int(frm[0]), int(frm[1]))
                except Exception as e:
                    warnings.warn(f'Did not understand range {frm},'
                                  f' got Exception {e}. \n', UserWarning)

                keys = [k + [slice(None), _rng] for k in keys]

            # A few metrics are special and calculated by ConditionArray
            # Assume only one key and user specified kwargs
            if name == 'predict_peaks':
                array.predict_peaks(keys[0], propagate=prop, **kwargs)
                return
            elif name in ('active_cells', 'active', 'cumulative_active'):
                array.mark_active_cells(keys[0], propagate=prop, **kwargs)
                return

            # Get the data and group with the key
            propagated = False
            arrs = [array[tuple(k)] for k in keys]
            arr_groups = itertools.permutations(zip(keys, arrs))
            func = getattr(np, func)
            for arrgrp in arr_groups:
                keys, arrs = zip(*arrgrp)
                result = func(*arrs, *args, **kwargs)

                # Each result is saved with the first key
                save_key = [name] + [k for k in keys[0]
                                     if not isinstance(k, slice)
                                     and k not in self._metric_idx]
                try:
                    array[tuple(save_key)] = result
                except ValueError as e:
                    '''This is super hackish, but need to
                    differentiate broadcasting error from
                    error due to incorrect array/result type'''
                    if 'could not broadcast' in e.args[0]:
                        result = np.expand_dims(result, axis=-1)
                        array[tuple(save_key)] = result
                    else:
                        raise ValueError(e)

                # Propagate results to other keys
                if prop and not propagated:
                    array.propagate_values(tuple(save_key), prop_to=prop)
                    propagated = True
                # If not including inverses, skip all remaining
                if not incl:
                    break

    def _apply_filters(self, array: ConditionArray) -> None:
        """Removes cells based on user-defined filters"""
        #for name, (metr, reg, chn, frm, args, kws) in self._filters.items():
        for f in self._filters:
            self.logger.info(f"Removing cells with filter {f['filter_name']}")
            self.logger.info(f"Inputs: {[f['region'], f['channel'], f['metric'], f['args'], f['kwargs']]}")
            self.logger.info(f"Current array size: {array.shape}")
            mask = array.generate_mask(f['filter_name'], f['metric'],
                                       f['region'], f['channel'],
                                       f['frame_rng'],
                                       *f['args'], **f['kwargs'])
            array.filter_cells(mask, delete=True)
            self.logger.info(f"Post-filter array size: {array.shape}")

    def _mark_skip_frames(self, skip_frames: Tuple[int] = None) -> None:
        """Marks any frames that should not be included

        Args:
            skip_frames: The frames to be skipped

        Returns:
            None
        """
        if skip_frames:
            self.logger.info(f'Marking bad frames {skip_frames}')
            # Mark the frames in the kwargs that are passed to extract_data
            new_kwargs = deepcopy(self.functions[-1][-1])
            new_kwargs['skip_frames'] = skip_frames

            # Functions are tuple, so make a copy to use
            new_func = [*self.functions[-1]]
            new_func[-1] = new_kwargs
            self.functions = [tuple(new_func)]

    def add_function(self,
                     func: str, *,
                     output_type: str = None,
                     name: str = None,
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
        kwargs = self.functions[0][-1]

        # Log inputs to Extract
        self.logger.info(f"Channels: {list(zip(kwargs['channels'], self.images))}")
        self.logger.info(f"Regions: {list(zip(kwargs['regions'], self.masks))}")
        self.logger.info(f"Metrics: {self._metrics}")
        self.logger.info(f"Added metrics: {list(self._props_to_add.keys())}")
        self.logger.info(f"Condition: {kwargs['condition']}")
        self.logger.info(f"Skipped frames: {kwargs['skip_frames']}")

        return super().run_operation(inputs)


class BaseEvaluate(Operation):
    """
    Base class for evaluation operations. Not currently implemented, but
    will be used to generate plots and metrics regarding the data from
    Extract.
    """
    __name__ = 'Evaluate'
    _input_type = (Array,)
    _output_type = Array

    def __init__(self,
                 output: str = 'evaluate',
                 **kwargs
                 ) -> None:
        super().__init__(output=output, **kwargs)
