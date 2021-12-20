import os
import sys
import argparse
import yaml
import warnings
import time
from typing import Dict, List, Collection, Tuple

import numpy as np
import tifffile as tiff
import imageio as iio

from cellst.operation import Operation
from cellst.utils._types import Image, Mask, Track, Arr, INPT_NAMES
from cellst.utils.process_utils import condense_operations, extract_operations
from cellst.utils.utils import folder_name
from cellst.utils.log_utils import get_logger, get_console_logger
from cellst.utils.yaml_utils import save_operation_yaml, save_pipeline_yaml

"""
TODO: Orchestrator and Pipeline might need to be in separate modules so they can
be called from the command line easily
"""

class Pipeline():
    """
    TODO:
        - file_location should be dir that pipeline was called from
        - Add calling from the command line
    """
    file_location = os.path.dirname(os.path.realpath(__file__))
    __name__ = 'Pipeline'
    __slots__ = ('_image_container', 'operations',
                 'parent_folder', 'output_folder',
                 'image_folder', 'mask_folder',
                 'track_folder', 'array_folder',
                 'operation_index', 'file_extension',
                 'logger', 'timer', 'overwrite',
                 'name', 'log_file')

    def __init__(self,
                 parent_folder: str = None,
                 output_folder: str = None,
                 image_folder: str = None,
                 mask_folder: str = None,
                 track_folder: str = None,
                 array_folder: str = None,
                 name: str = None,
                 file_extension: str = 'tif',
                 overwrite: bool = True,
                 log_file: bool = True
                 ) -> None:
        """
        Pipeline will only handle a folder that has images in it.
        This could be changed based on the regex (like if each folder has a different channel)
        But otherwise, multiple folders need to be handled by the Orchestrator

        Args:
            image_folder (str) = relative path to the folder containing the images
            channels (List[str]) = List of identifiers for each channel

        Returns:
            None
        """
        # Save some values in case Pipeline is written as yaml
        self.overwrite = overwrite
        self.log_file = log_file

        # Define paths to find and save images
        self.file_extension = file_extension
        self._set_all_paths(parent_folder, output_folder, image_folder,
                            mask_folder, track_folder, array_folder)
        self._make_output_folder(overwrite)
        self.name = folder_name(self.parent_folder) if name is None else name

        # Set up logger - defaults to output folder
        if log_file:
            self.logger = get_logger(self.__name__, self.output_folder,
                                     overwrite=overwrite)
        else:
            self.logger = get_console_logger()

        # Prepare for getting operations and images
        self._image_container = {}
        self.operations = []

        # Log relevant information and parameters
        self.logger.info(f'Pipeline {repr(self)} initiated.')
        self.logger.info(f'Parent folder: {self.parent_folder}')
        self.logger.info(f'Output folder: {self.output_folder}')
        self.logger.info(f'Image folder: {self.image_folder}')
        self.logger.info(f'Mask folder: {self.mask_folder}')
        self.logger.info(f'Track folder: {self.track_folder}')
        self.logger.info(f'Array folder: {self.array_folder}')

    def __enter__(self) -> None:
        """
        """
        # Create the image container if needed
        if not hasattr(self, '_image_container'):
            self._image_container = {}

        # Start a timer
        self.timer = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        """
        # Remove image container from memory
        self._image_container = None
        del self._image_container

        # Log time spent after enter
        try:
            self.logger.info(f'Total execution time: {time.time() - self.timer}')
            self.timer = None
        except TypeError:
            # KeyboardInterrupt now won't cause additional exceptions
            pass

    def __str__(self) -> str:
        """
        """
        # Inputs are printed first
        string = self.parent_folder
        _fols = ['image', 'mask', 'track', 'array']
        for imtyp in _fols:
            string += (f'\n\t {imtyp}: ' +
                       folder_name(getattr(self, f'{imtyp}_folder')))

        # Add operations to string
        for oper in self.operations:
            string += f'\n {oper}'

        # Outputs last
        string += f'\n {self.output_folder}'

        return string

    def add_operations(self,
                       operation: Collection[Operation],
                       index: int = -1
                       ) -> None:
        """
        Adds Operations to the Pipeline and recalculates the index

        Args:

        Returns:

        TODO:
            - Not sure the index results always make sense, best to add all
              operations at once
        """
        # Adds operations to self.operations (Collection[Operation])
        if isinstance(operation, Collection):
            if all([isinstance(o, Operation) for o in operation]):
                if index == -1:
                    self.operations.extend(operation)
                else:
                    self.operations[index:index] = operation
            else:
                raise ValueError('Not all elements of operation are class Operation.')
        elif isinstance(operation, Operation):
            if index == -1:
                self.operations.append(operation)
            else:
                self.operations.insert(index, operation)
        else:
            raise ValueError(f'Expected type Operation, got {type(operation)}.')

        # Log changes to the list of operations
        self.logger.info(f'Added {len(operation)} operations at {index}. '
                         f'\nCurrent operation list: {self.operations}.')

        self.operation_index = {i: o for i, o in enumerate(self.operations)}

    def run(self) -> (Image, Mask, Track, Arr):
        """
        Runs all of the operations in self.operations on the images

        Args:

        Returns:
        """
        # Can skip doing anything if no operations have been added
        if len(self.operations) == 0:
            warnings.warn('No operations in Pipeline. Returning None.',
                          UserWarning)
            return

        # This will delete images after finished running
        with self:
            # Determine needed inputs and outputs and load to container
            inputs, outputs = self._input_output_handler()
            self._load_images_to_container(self._image_container, inputs, outputs)

            for inpts, otpts, oper in zip(inputs, outputs, self.operations):
                # Log the operation
                self.logger.info(str(oper))

                # Get images and pass to operation
                try:
                    imgs, msks, trks, arrs = self._get_images_from_container(inpts)
                except KeyError as e:
                    raise KeyError(f'Failed to find all inputs for {oper} \n',
                                   e.__str__())

                # Run the operation and save results
                oper.set_logger(self.logger)
                with oper:
                    oper_result = oper.run_operation(imgs, msks, trks, arrs)
                    self._image_container = self.update_image_container(
                                                self._image_container,
                                                oper_result,
                                                otpts)

                    # Write to disk if needed
                    if oper.save:
                        self.save_images(oper.save_arrays,
                                         oper._output_type.__name__)

        return oper_result

    def update_image_container(self,
                               container: Dict[str, np.ndarray],
                               array: (Image, Mask, Track),
                               key: Tuple[str],
                               ) -> None:
        """
        """
        if key not in container:
            container[key] = array
        else:
            # TODO: Should there be an option to not overwrite?
            container[key] = array

        return container

    def save_images(self,
                    save_arrays: Dict[str, Tuple],
                    oper_output: str = None,
                    img_dtype: type = None
                    ) -> None:
        """
        New save function to handle multiple saves per operation

        TODO:
            - Test different iteration strategies for efficiency
            - Should not upscale images
            - Use type instead of str for output (if even needed)
            - Allow for non-consecutive indices (how?)
            - There is no way to pass dtype to this function currently
        """
        for name, (otpt_type, arr) in save_arrays.items():
            # Make output directory if needed
            save_folder = os.path.join(self.output_folder, name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Save CellArray separately
            if oper_output == 'array':
                name = os.path.join(self.output_folder, f"{otpt_type}.hdf5")
                arr.save(name)

                self.logger.info(f'Saved data frame in {self.output_folder}. '
                                 f'shape: {arr.shape}, type: {arr.dtype}.')
            else:
                save_dtype = arr.dtype if img_dtype is None else img_dtype
                if arr.ndim != 3:
                    warnings.warn("Expected stack with 3 dimensions."
                                  f"Got {arr.ndim} for {name}", UserWarning)

                # Set length of index based on total number of frames
                zrs = len(str(arr.shape[0]))
                # Save files as tiff with consecutive idx
                for idx in range(arr.shape[0]):
                    name = os.path.join(save_folder, f"{otpt_type}{idx:0{zrs}}.tiff")
                    tiff.imsave(name, arr[idx, ...].astype(save_dtype))

                self.logger.info(f'Saved {arr.shape[0]} images in {save_folder}.')

    def save_as_yaml(self,
                     path: str = None,
                     fname: str = None
                     ) -> None:
        """
        Should write Pipeline as a yaml file to output dir
        """
        # Set path for saving files - saves in output by default
        path = self.output_folder if path is None else path
        if fname is None:
            fname = f'{folder_name(self.parent_folder)}.yaml'
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, fname)

        # Create specification as dictionary
        pipe_dict = self._pipeline_to_dict()

        self.logger.info(f'Saving Pipeline {repr(self)} in {path}.')
        save_pipeline_yaml(path, pipe_dict)

    def save_operations_as_yaml(self,
                                path: str = None,
                                fname: str = 'operations.yaml'
                                ) -> None:
        """
        Save operations as a stand-alone yaml file.
        """
        # Get the path
        path = self.output_folder if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, fname)

        # Save using yaml_utils
        self.logger.info(f"Saving Operations at {path}")
        save_operation_yaml(path, self.operations)

    @classmethod
    def load_from_yaml(cls, path: str) -> 'Pipeline':
        """Builds Pipeline class from specifications in yaml file"""
        with open(path, 'r') as yf:
            pipe_dict = yaml.load(yf, Loader=yaml.FullLoader)

        return cls._build_from_dict(pipe_dict)

    def _input_output_handler(self) -> Tuple[str]:
        """
        Iterate through all the operations and figure out which images
        to save and which to remove.

        TODO:
            - Determine after which operation the stack is no longer needed
        """
        # Inputs and outputs determined by the args passed to each Operation
        req_inputs = []
        req_outputs = []
        for o in self.operations:
            imgs = [tuple([i, Image.__name__]) for i in o.input_images]
            msks = [tuple([m, Mask.__name__]) for m in o.input_masks]
            trks = [tuple([t, Track.__name__]) for t in o.input_tracks]
            arrs = [tuple([a, Arr.__name__]) for a in o.input_arrays]

            req_inputs.append([imgs, msks, trks, arrs])
            req_outputs.append(o.output_id)

        # Log the inputs and outputs
        self.logger.info('Expected images: '
                         f'{[i[0] for op in req_inputs for i in op[0]]}')
        self.logger.info('Expected masks: '
                         f'{[i[0] for op in req_inputs for i in op[1]]}')
        self.logger.info('Expected tracks: '
                         f'{[i[0] for op in req_inputs for i in op[2]]}')
        self.logger.info('Expected arrays: '
                         f'{[i[0] for op in req_inputs for i in op[3]]}')
        self.logger.info(f'Exected outputs: {[r[0] for r in req_outputs]}')

        return req_inputs, req_outputs

    def _get_image_paths(self,
                         folder: str,
                         match_str: str,
                         subfolder: str = None
                         ) -> Collection[str]:
        """
        match_str: Tuple[str], [0] is the  mat

        1. Look for images in subfolder if given
        2. Look for images in folder
        3. Look for subfolder that matches match_str.
        4. If exists, return those images.

        TODO:
            - Add image selection based on regex
            - Should channels use regex or glob to match name
            - Could be moved to a utils file - staticmethod
            - Include a check for file type as well!!
                - Yeah, they are all over the place here. Need to be in _load
                  as well
        """
        # First check in designated subfolder
        if subfolder is not None:
            folder = os.path.join(folder, subfolder)

        # Function to check if img should be loaded
        def _confirm_im_match(im, check_name: bool = True) -> bool:
            name = True if not check_name else match_str in im
            ext = self.file_extension in im
            fil = os.path.isfile(os.path.join(folder, im))

            return name * ext * fil

        # Find images and sort into list
        im_names = [os.path.join(folder, im)
                    for im in sorted(os.listdir(folder))
                    if _confirm_im_match(im)]

        # If no images were found, look for a subdirectory
        if len(im_names) == 0:
            try:
                # Take first subdirectory found that has match_str
                subfol = [fol for fol in sorted(os.listdir(folder))
                          if os.path.isdir(os.path.join(folder, fol))
                          if match_str in fol][0]
                folder = os.path.join(folder, subfol)

                # Look ONLY for images in that subdirectory
                # Load all images, even if match_str doesn't match
                im_names = [os.path.join(folder, im)
                            for im in sorted(os.listdir(folder))
                            if _confirm_im_match(im, check_name=False)]
            except IndexError:
                # Indidcates that no sub_folders were found
                # im_names should be [] at this point
                pass

        return im_names

    def _load_images_to_container(self,
                                  container: Dict[tuple, np.ndarray],
                                  inputs: Collection[tuple],
                                  outputs: Collection[tuple],
                                  img_dtype: type = np.int16
                                  ) -> Dict[tuple, np.ndarray]:
        """
        Loads image files and saves the result as a 3D np.ndarray
        in the given container.
        If only one image is found, the dimension of the array is
        expanded along the second axis. (and first if needed)
        Image type defaults to 16bit.
            Uses int instead of uint to allow for negative values
        It is assumed that channel_images are already sorted

        Args:

        Returns:

        TODO:
            - Add saving of image metadata
            - No way to pass img_dtype to this function
        """
        for idx, name in enumerate(INPT_NAMES):
            # Get unique list of all inputs requested by operations
            all_requested = [sl for l in [i[idx] for i in inputs] for sl in l]
            to_load = list(set(all_requested))

            for key in to_load:
                fol = getattr(self, f'{name}_folder')
                pths = self._get_image_paths(fol, key[0])
                if len(pths) == 0:
                    # If no images are found in the path, check output_folder
                    fol = self.output_folder
                    pths = self._get_image_paths(fol, key[0])
                    # If still no images, check the listed outputs
                    if len(pths) == 0 and key not in outputs:
                        # TODO: The order matters. Should raise error if it is made
                        #       after it is needed, otherwise continue.
                        raise ValueError(f'Data {key} cannot be found and is '
                                         'not listed as an output.')

                # Log the paths
                self.logger.info(f'Looking for {key[0]} in {fol}. '
                                 f'Found {len(pths)} files.')

                # Load the images
                '''NOTE: Using mimread instead of imread to add the option of limiting
                the memory of the loaded image. However, mimread still only loads one
                file at a time, so it is unlikely to ever hit the memory limit.
                Could be worth tracking the memory and deleting large arrays or
                temporarily storing them in a file (if low-mem mode is true)
                Slightly faster than imread.'''
                # Pre-allocate numpy array for speed
                for n, p in enumerate(pths):
                    img = iio.mimread(p)[0]

                    if n == 0:
                        # Initialize img_stack if it doesn't exist
                        img_stack = np.empty(tuple([len(pths), img.shape[0], img.shape[1]]),
                                             dtype=img.dtype)

                    img_stack[n, ...] = img

                container[key] = img_stack

                self.logger.info(f'Images loaded. shape: {img_stack.shape}, '
                                 f'type: {img_stack.dtype}.')

        return container

    def _get_images_from_container(self,
                                   input_keys: Collection[Collection],
                                   ) -> List[np.ndarray]:
        """
        Checks for key in the image container and returns the corresponding stacks
        Raises error if not found.

        TODO:
            - If it doesn't find a key, could first try reloading the _image_container
        """
        temp = []
        for keys in input_keys:
            try:
                temp.append(
                    [self._image_container[k] for k in keys]
                )
            except KeyError:
                raise KeyError(f'Image stack {keys} does not exist.')

        return temp

    def _set_all_paths(self,
                       parent_folder: str,
                       output_folder: str,
                       image_folder: str,
                       mask_folder: str,
                       track_folder: str,
                       array_folder: str
                       ) -> None:
        """
        TODO:
            - Should these use Path module?
            - Should image, mask, track be set as relative folders?
                - If so, based on outputs or on parent?
        """
        # Parent folder defaults to folder where Pipeline was called
        if parent_folder is None:
            self.parent_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
        else:
            self.parent_folder = parent_folder

        # Output folder defaults to folder in parent_folder
        if output_folder is None:
            self.output_folder = os.path.join(self.parent_folder, 'outputs')
        else:
            self.output_folder = output_folder

        # Image folder defaults parent_folder
        if image_folder is None:
            self.image_folder = self.parent_folder
        else:
            self.image_folder = os.path.abspath(image_folder)

        # All others default to output folder
        _fols = ['mask', 'track', 'array']
        self.mask_folder = mask_folder
        self.track_folder = track_folder
        self.array_folder = array_folder

        for imtyp in _fols:
            if getattr(self, f'{imtyp}_folder') is None:
                setattr(self, f'{imtyp}_folder', self.output_folder)

    def _make_output_folder(self,
                            overwrite: bool = True
                            ) -> None:
        """
        This will also be responsible for logging and passing the yaml
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            fmode = 'a'
        elif overwrite:
            fmode = 'w'
        elif not overwrite:
            fmode = 'a'
            it = 0
            tempdir = self.output_folder + f'_{it}'
            while os.path.exists(tempdir):
                it += 1
                tempdir = self.output_folder + f'_{it}'

            self.output_folder = tempdir
            os.makedirs(self.output_folder)

    def _pipeline_to_dict(self) -> Dict:
        """
        Saves the Pipeline as a dictionary to put into yaml file
        """
        # Save basic parameters
        init_params = ('parent_folder', 'output_folder', 'image_folder',
                       'mask_folder', 'track_folder', 'array_folder',
                       'overwrite', 'file_extension', 'log_file', 'name')
        pipe_dict = {att: getattr(self, att) for att in init_params}

        # Add Operations to dict
        if self.operations:
            pipe_dict['_operations'] = condense_operations(self.operations)

        return pipe_dict

    def _load_operations_from_dict(self,
                                   oper_dict: Dict[str, Operation]
                                   ) -> None:
        """Adds operations to self from a dictionary of Operations i.e. yaml"""
        self.add_operations(extract_operations(oper_dict))

    @classmethod
    def _build_from_dict(cls, pipe_dict: dict) -> 'Pipeline':
        # Save Operations to use later
        try:
            op_dict = pipe_dict.pop('_operations')
        except KeyError:
            # Means no Operations
            op_dict = {}

        # Load pipeline
        pipe = cls(**pipe_dict)

        # Save condition to add to Extract operations
        # TODO: This won't work for multiple Extract operations
        condition = pipe_dict['name']
        if 'extract' in op_dict and op_dict['extract']['condition'] == 'default':
            op_dict['extract']['condition'] = condition

        # Load the operations
        pipe._load_operations_from_dict(op_dict)

        return pipe

    @classmethod
    def _run_single_pipe(cls,
                         pipe_dict: Dict
                         ) -> Arr:
        """
        Creates a Pipeline object, adds operations, and runs.

        NOTE: This might be overkill, but don't want it to persist
              in memory.
        NOTE: Assumes it is okay to make changes in-place in oper,
              i.e. that it is a copy of the original dictionary

        TODO: This should, from now on, only take in Pipe. Everything
              else should be handled externally.
        """
        pipe = cls._build_from_dict(pipe_dict)
        with pipe:
            result = pipe.run()

        # Remove from memory
        del pipe

        return result
