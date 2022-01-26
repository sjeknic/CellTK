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
from cellst.utils._types import Image, Mask, Track, Arr, ImageContainer
from cellst.utils.process_utils import condense_operations, extract_operations
from cellst.utils.log_utils import get_logger, get_console_logger
from cellst.utils.file_utils import (save_operation_yaml, save_pipeline_yaml,
                                     folder_name)


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
                 'name', 'log_file', '_split_key',
                 'completed_ops', '__dict__')

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
                 log_file: bool = True,
                 yaml_path: str = None,
                 verbose: bool = False,
                 _split_key: str = '&'
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
        self._split_key = _split_key

        # Define paths to find and save images
        self.file_extension = file_extension
        self._set_all_paths(parent_folder, output_folder, image_folder,
                            mask_folder, track_folder, array_folder)
        self._make_output_folder(overwrite)
        self.name = folder_name(self.parent_folder) if name is None else name

        # Set up logger - defaults to output folder
        if log_file:
            lev = 'info' if verbose else 'warning'
            self.logger = get_logger(self.__name__, self.output_folder,
                                     overwrite=overwrite, console_level=lev)
        else:
            self.logger = get_console_logger()

        # Copy yaml file to output
        # if yaml_path:
        #     targ = os.path.join(self.output_folder, yaml_path.split('/')[-1])
        #     shutil.move(yaml_path, targ)
        #     self.logger.info(f'Moved input yaml to: {targ}')

        # Prepare for getting operations and images
        self._image_container = ImageContainer()
        self.operations = []

        # Log relevant information and parameters
        self.logger.info(f'Pipeline initiated.')
        self.logger.info(f'Pipeline: {repr(self)}.')
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
            self._image_container = ImageContainer()

        # Start a timer
        self.timer = time.time()
        self.completed_ops = 0

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

        # Log if pipeline was completed
        if self.completed_ops == len(self.operations):
            self.logger.info('Pipeline completed.')
        else:
            self.logger.info(f'{self.completed_ops} / {len(self.operations)} '
                             'operations completed by Pipeline.')

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
        # TODO: Dirty fix - Need to fix this in the master branch
        if isinstance(operation, Operation):
            operation = [operation]

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
                    imgs_for_operation = self._get_images_from_container(inpts)
                except KeyError as e:
                    raise KeyError(f'Failed to find all inputs for {oper} \n',
                                   e.__str__())

                # Run the operation and save results
                oper.set_logger(self.logger)
                with oper:
                    # oper_result is a generator for the results and keys
                    oper_result = oper.run_operation(imgs_for_operation)
                    self._image_container.update(dict(oper_result))
                    # Write to disk if needed
                    self.save_images(oper.save_arrays,
                                     oper._output_type.__name__)

                self.completed_ops += 1

        return oper_result

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
            save_folder = os.path.join(self.output_folder, name)

            # Save CellArray separately
            if oper_output == 'array':
                name = os.path.join(self.output_folder, f"{name}.hdf5")
                arr.save(name)

                self.logger.info(f'Saved data frame at {name}. '
                                 f'shape: {arr.shape}, type: {arr.dtype}.')
            else:
                # Make output directory if needed
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

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

        # Save using file_utils
        self.logger.info(f"Saving Operations at {path}")
        save_operation_yaml(path, self.operations)

    @classmethod
    def load_from_yaml(cls, path: str) -> 'Pipeline':
        """Builds Pipeline class from specifications in yaml file"""
        with open(path, 'r') as yf:
            pipe_dict = yaml.load(yf, Loader=yaml.Loader)

        return cls._build_from_dict(pipe_dict, path)

    def _input_output_handler(self) -> List[List[Tuple[str]]]:
        """
        Iterate through all the operations and figure out which images
        to save and which to remove.

        Returns:
            Output =

        TODO:
            - Determine after which operation the stack is no longer needed
            - Call function to delete uneeded stacks (probably after oper.__exit__)
        """
        # Inputs and outputs determined by the args passed to each Operation
        req_inputs = []
        req_outputs = []
        for o in self.operations:
            op_in, op_out = o.get_inputs_and_outputs()
            req_inputs.append(op_in)
            req_outputs.append(op_out)

        # Log the inputs and outputs
        self.logger.info(f'Expected inputs: {req_inputs}')
        self.logger.info(f'Exected outputs: {req_outputs}')

        return req_inputs, req_outputs

    def _get_image_paths(self,
                         folder: str,
                         key: Tuple[str],
                         ) -> Collection[str]:
        """
        Steps to get images:
        1. Check folder for images that match
        2. Check subfolders for one that matches match_str EXACTLY
            2a. Load images containing match_str
            2b. Load images that match type by name
            2c. Load all the images

        TODO:
            - Add image selection based on regex
            - Should channels use regex or glob to match name
        """
        match_str, im_type = key

        # Check if key includes an operation identifier
        if self._split_key in match_str:
            fol_id, match_str = match_str.split(self._split_key)
            folder = os.path.join(folder, fol_id)

        self.logger.info(f'Looking for {match_str} of type '
                         f'{im_type} in {folder}.')

        # Function to check if img should be loaded
        def _confirm_im_match(im: str, match_str: str, path: str) -> bool:
            name = True if match_str is None else match_str in im
            ext = self.file_extension in im
            fil = os.path.isfile(os.path.join(path, im))

            return name * ext * fil

        # Walk through all directories in folder
        im_names = []  # Needed in case walk doesn't return anything
        for lvl, (pth, dirs, files) in enumerate(os.walk(folder)):
            if lvl == 0:
                # Check for images that match in first folder
                im_names = [os.path.join(pth, f) for f in sorted(files)
                            if _confirm_im_match(f, match_str, pth)]

            elif pth.split('/')[-1] == match_str:
                # Check for images if in dir that matches
                for st in (match_str, im_type, None):
                    im_names = [os.path.join(pth, f) for f in sorted(files)
                                if _confirm_im_match(f, st, pth)]
                    if im_names: break

            if im_names: break

        self.logger.info(f'Found {len(im_names)} images.')
        return im_names

    def _load_images_to_container(self,
                                  container: ImageContainer,
                                  inputs: Collection[tuple],
                                  outputs: Collection[tuple],
                                  img_dtype: type = np.int16
                                  ) -> ImageContainer:
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
        # Get list of all outputs/unique inputs from operations
        outputs = [sl for l in outputs for sl in l]
        all_requested = [sl for l in inputs for sl in l]
        to_load = list(set(all_requested))

        for key in to_load:
            fol = getattr(self, f'{key[1]}_folder')
            pths = self._get_image_paths(fol, key)

            if not pths:
                # If no images are found in the path, check output_folder
                if fol != self.output_folder:
                    pths = self._get_image_paths(self.output_folder, key)

                # If still no images, check the listed outputs
                if not pths:
                    # Accept outputs that include save_name, but not channel
                    if self._split_key in key[0]:
                        _out = key[0].split(self._split_key)[0]
                        _out = (_out, key[1])
                        if _out in outputs: continue
                    elif key not in outputs:
                        # If nothing, check for whole key in outputs
                        # TODO: Include consideration of operation order
                        raise ValueError(f'Data {key} cannot be found and is '
                                         'not listed as an output.')

            # TODO: This check is redundant, simplify later
            if pths:
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

                # Make img_stack read-only. To change image stack, overwrite container[key]
                img_stack.flags.writeable = False
                container[key] = img_stack

                self.logger.info(f'Images loaded. shape: {img_stack.shape}, '
                                 f'type: {img_stack.dtype}.')

        return container

    def _get_images_from_container(self,
                                   input_keys: Collection[Collection],
                                   ) -> ImageContainer:
        """
        Checks for key in the image container and returns the corresponding stacks
        Raises error if not found.

        TODO:
            - If it doesn't find a key, could first try reloading the _image_container
        """
        # Create a copy that points to the same locations in memory
        new_container = ImageContainer()
        # TODO: Should a KeyError be caught here?
        new_container.update({k: self._image_container[k] for k in input_keys
                              if k in self._image_container})

        return new_container

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
                       'overwrite', 'file_extension', 'log_file', 'name',
                       '_split_key')
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
    def _build_from_dict(cls, pipe_dict: dict, yaml_path: str = None) -> 'Pipeline':
        # Save Operations to use later
        try:
            op_dict = pipe_dict.pop('_operations')
        except KeyError:
            # Means no Operations
            op_dict = {}

        # Load pipeline
        pipe = cls(**pipe_dict, yaml_path=yaml_path)

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
        result = pipe.run()

        # Remove from memory
        del pipe

        return result

    @classmethod
    def _get_command_line_inputs(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        # Input file is easiest way to pass arguments
        parser.add_argument(
            '--yaml', '-y',
            default=None,
            type=str,
            help='YAML file containing input parameters'
        )

        # Consider whether to add these
        parser.add_argument(
            '--output',
            type=str,
            default='output',
            help='Name of output directory. Default is output.'
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='If set, will overwrite contents of output folder. Otherwise makes new folder.'
        )

        return parser.parse_args()


if __name__ == '__main__':
    args = Pipeline._get_command_line_inputs()

    yaml_path = args.yaml
    pipe = Pipeline.load_from_yaml(yaml_path)
    pipe.run()
