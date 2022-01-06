import os
import sys
import argparse
import yaml
import warnings
import time
from multiprocessing import Pool
from typing import Dict, List, Collection, Tuple
from glob import glob

import numpy as np
import tifffile as tiff
import imageio as iio

from cellst.operation import Operation
from cellst.utils._types import Image, Mask, Track, Arr, ImageContainer
from cellst.utils._types import Experiment
from cellst.utils.process_utils import condense_operations, extract_operations
from cellst.utils.utils import folder_name
from cellst.utils.log_utils import get_logger, get_console_logger
from cellst.utils.yaml_utils import save_operation_yaml, save_pipeline_yaml


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
                 'name', 'log_file', '_split_key')

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
            self.logger = get_logger(self.__name__, self.output_folder,
                                     overwrite=overwrite)
        else:
            self.logger = get_console_logger()

        # Prepare for getting operations and images
        self._image_container = ImageContainer()
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
            self._image_container = ImageContainer()

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
                    if oper.save:
                        self.save_images(oper.save_arrays,
                                         oper._output_type.__name__)

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
            # Make output directory if needed
            save_folder = os.path.join(self.output_folder, name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Save CellArray separately
            if oper_output == 'array':
                name = os.path.join(self.output_folder, f"{name}.hdf5")
                arr.save(name)

                self.logger.info(f'Saved data frame at {name}. '
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
        def _confirm_im_match(im: str, match_str: str) -> bool:
            name = True if match_str is None else match_str in im
            ext = self.file_extension in im
            fil = os.path.isfile(os.path.join(folder, im))

            return name * ext * fil

        # Walk through all directories in folder
        im_names = []  # Needed in case walk doesn't return anything
        for lvl, (pth, dirs, files) in enumerate(os.walk(folder)):
            if lvl == 0:
                # Check for images that match in first folder
                im_names = [os.path.join(pth, f) for f in sorted(files)
                            if _confirm_im_match(f, match_str)]

            elif pth.split('/')[-1] == match_str:
                # Check for images if in dir that matches
                for st in (match_str, im_type, None):
                    im_names = [os.path.join(pth, f) for f in sorted(files)
                                if _confirm_im_match(f, st)]
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


class Orchestrator():
    """
    TODO:
        - Add __str__ method
        - Add function for submitting jobs to SLURM
    """
    file_location = os.path.dirname(os.path.realpath(__file__))

    __name__ = 'Orchestrator'
    __slots__ = ('pipelines', 'operations', 'yaml_folder',
                 'parent_folder', 'output_folder',
                 'operation_index', 'file_extension', 'name',
                 'overwrite', 'save', 'condition_map',
                 'logger')

    def __init__(self,
                 yaml_folder: str = None,
                 parent_folder: str = None,
                 output_folder: str = None,
                 match_str: str = None,
                 image_folder: str = None,
                 mask_folder: str = None,
                 track_folder: str = None,
                 array_folder: str = None,
                 condition_map: dict = {},
                 name: str = 'experiment',
                 file_extension: str = 'tif',
                 overwrite: bool = True,
                 log_file: bool = True,
                 save_master_df: bool = True,
                 ) -> None:
        """
        Args:

        Returns:
            None

        TODO:
            - Should be able to parse args and load a yaml as well
            - Should be able to load yaml to define operations
        """
        # Save some values
        self.name = name
        self.file_extension = file_extension
        self.overwrite = overwrite
        self.save = save_master_df
        self.condition_map = condition_map

        # Set paths and start logging
        self._set_all_paths(yaml_folder, parent_folder, output_folder)
        self._make_output_folder(self.overwrite)
        if log_file:
            self.logger = get_logger(self.__name__, self.output_folder,
                                     overwrite=overwrite)
        else:
            self.logger = get_console_logger()

        # Build the Pipelines and input/output paths
        self.pipelines = {}
        self._build_pipelines(match_str, image_folder, mask_folder,
                              track_folder, array_folder)


        # Prepare for getting operations
        self.operations = []

    def __len__(self) -> int:
        return len(self.pipelines)

    def run(self, n_cores: int = 1) -> None:
        """
        Run all the Pipelines with all of the operations.

        TODO:
            - Should results really be saved?
        """
        # Can skip doing anything if no operations have been added
        if len(self.operations) == 0 or len(self.pipelines) == 0:
            warnings.warn('No Operations and/or Pipelines. Returning None.',
                          UserWarning)
            return

        # Add operations to the pipelines
        self.add_operations_to_pipelines(self.operations)

        # Run with multiple cores or just a single core
        if n_cores > 1:
            results = self.run_multiple_pipelines(self.pipelines,
                                                  n_cores=n_cores)
        else:
            results = []
            for fol, kwargs in self.pipelines.items():
                results.append(Pipeline._run_single_pipe(kwargs))

        if self.save:
            # TODO: If results are saved, pass here, otherwise, use files
            self.build_experiment_file()

        return results

    def run_multiple_pipelines(self,
                               pipeline_dict: Dict,
                               n_cores: int = 1
                               ) -> Collection[Arr]:
        """
        pipeline_dict holds path information for building ALL of the pipelines
            - key is subfolder, val is to be passed to Pipeline.__init__

        Assumes that operations are already added to the Pipelines
        """
        # Run asynchronously
        with Pool(n_cores) as pool:
            # Set up pool of workers
            multi = [pool.apply_async(Pipeline._run_single_pipe, args=(kwargs))
                     for kwargs in pipeline_dict.values()]

            # Run pool and return results
            return [m.get() for m in multi]

    def add_operations(self,
                       operation: Collection[Operation],
                       index: int = -1
                       ) -> None:
        """
        Adds Operations to the Orchestrator and recalculates the index

        Args:

        Returns:
        """
        if isinstance(operation, Collection):
            if all([isinstance(o, Operation) for o in operation]):
                if index == -1:
                    self.operations.extend(operation)
                else:
                    self.operations[index:index] = operation
            else:
                raise ValueError('Not all items in operation are Operations.')
        elif isinstance(operation, Operation):
            if index == -1:
                self.operations.append(operation)
            else:
                self.operations.insert(index, operation)
        else:
            raise ValueError(f'Expected Operation, got {type(operation)}.')

        self.operation_index = {i: o for i, o in enumerate(self.operations)}

    def build_experiment_file(self, arrays: Collection[Arr] = None) -> None:
        """
        Search folders in self.pipelines for hdf5 data frames
        """
        # Make Experiment array to hold data
        out = Experiment(name=self.name)

        # Search for all dfs in all pipeline folders
        for fol in self.pipelines:
            otpt_fol = os.path.join(self.output_folder, fol)

            # NOTE: if df.name is already in Experiment, will be overwritten
            for df in glob(os.path.join(otpt_fol, '*.hdf5')):
                out.load_condition(df)

        # Save the master df file
        save_path = os.path.join(self.output_folder, f'{self.name}.hdf5')
        out.save(save_path)

    def add_operations_to_pipelines(self,
                                    operations: Collection[Operation] = []
                                    ) -> None:
        """
        Adds Operations to each Pipeline in Orchestrator
        """
        # Collect operations
        op_dict = condense_operations(operations)

        self.logger.info(f'Adding Operations {operations} '
                         f'to {len(self)} Pipelines.')
        for pipe, kwargs in self.pipelines.items():
            # First try to append operations before overwriting
            try:
                kwargs['_operations'].update(op_dict)
            except KeyError:
                kwargs.update({'_operations': op_dict})

    def update_condition_map(self, condition_map: dict = {}) -> None:
        """
        Adds conditions to each of the Pipelines in Orchestrator
        """
        for fol, cond in condition_map.items():
            self.pipelines[fol]['name'] = cond

        self.condition_map = condition_map

    def save_pipelines_as_yamls(self, path: str = None) -> None:
        """
        Save yaml file that can be loaded as Pipeline
        """
        # Set path for saving files - saves in yaml folder
        path = self.output_folder if path is None else path
        path = os.path.join(path, 'pipeline_yamls')
        if not os.path.exists(path):
            os.makedirs(path)

        # Make sure operations are added to pipelines
        self.add_operations_to_pipelines(self.operations)

        # Save each pipeline
        self.logger.info(f"Saving {len(self.pipelines)} yamls in {path}")
        for pipe, kwargs in self.pipelines.items():
            # Save the Pipeline
            fname = os.path.join(path,
                                 f"{folder_name(kwargs['parent_folder'])}.yaml")
            save_pipeline_yaml(fname, kwargs)

    def save_operations_as_yaml(self,
                                path: str = None,
                                fname: str = 'operations.yaml'
                                ) -> None:
        """
        Save self.operations as a yaml file.
        """
        # Get the path
        path = self.output_folder if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, fname)

        # Save using yaml_utils
        self.logger.info(f"Saving Operations at {path}")
        save_operation_yaml(path, self.operations)

    def _build_pipelines(self,
                         match_str: str,
                         image_folder: str,
                         mask_folder: str,
                         track_folder: str,
                         array_folder: str
                         ) -> None:
        """
        """
        # If yamls are provided, use those to make the Pipelines
        if self.yaml_folder is not None:
            files = [os.path.join(self.yaml_folder, f)
                     for f in os.listdir(self.yaml_folder)
                     if f.endswith('.yaml')]
            self.logger.info(f'Found {len(files)} possible pipelines '
                             f'in {self.yaml_folder}')
            # Load all yamls as dictionaries
            for f in files:
                with open(f, 'r') as yf:
                    pipe = yaml.load(yf, Loader=yaml.FullLoader)
                    try:
                        fol = folder_name(pipe['parent_folder'])
                        self.pipelines[fol] = pipe
                    except KeyError:
                        # Indicates yaml file was not a Pipeline yaml
                        pass
        else:
            # Find all folders that will be needed for Pipelines
            self.logger.info(f'Found {len(os.listdir(self.parent_folder))} '
                             f'possible pipelines in {self.parent_folder}')
            for fol in os.listdir(self.parent_folder):
                # Check for the match_str
                if match_str is not None and match_str not in fol:
                    continue

                # Make sure the path is a directory
                fol_path = os.path.join(self.parent_folder, fol)
                if os.path.isdir(fol_path) and fol_path != self.output_folder:
                    # First initialize the dictionary
                    self.pipelines[fol] = {}

                    # Save parent folder
                    self.pipelines[fol]['parent_folder'] = fol_path

                    # Save output folder relative to self.output_folder
                    out_fol = os.path.join(self.output_folder, fol)
                    self.pipelines[fol]['output_folder'] = out_fol

                    # Set all of the subfolders
                    self.pipelines[fol].update(dict(
                        image_folder=self._set_rel_to_par(fol_path,
                                                          image_folder),
                        mask_folder=self._set_rel_to_par(fol_path,
                                                         mask_folder),
                        track_folder=self._set_rel_to_par(fol_path,
                                                          track_folder),
                        array_folder=self._set_rel_to_par(fol_path,
                                                          array_folder),
                    ))

                    # Add condition
                    try:
                        self.pipelines[fol]['name'] = self.condition_map[fol]
                    except KeyError:
                        self.pipelines[fol]['name'] = fol

                    # Add miscellaneous options
                    self.pipelines[fol]['file_extension'] = self.file_extension
                    self.pipelines[fol]['overwrite'] = self.overwrite

        self.logger.info(f'Loaded {len(self)} pipelines')

    def _set_all_paths(self,
                       yaml_folder: str,
                       parent_folder: str,
                       output_folder: str
                       ) -> None:
        # Parent folder defaults to folder where Orchestrator was called
        if parent_folder is None:
            self.parent_folder = os.path.abspath(sys.argv[0])
        else:
            self.parent_folder = parent_folder

        # Output folder defaults to folder in parent_folder
        if output_folder is None:
            self.output_folder = os.path.join(self.parent_folder, 'outputs')
        else:
            self.output_folder = output_folder

        # YAML folder can remain None
        self.yaml_folder = yaml_folder

    def _make_output_folder(self,
                            overwrite: bool = True
                            ) -> None:
        """
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

    def _set_rel_to_par(self,
                        par_path: str,
                        inpt: str
                        ) -> str:
        """
        Makes inpts relative to the parent_folder of that Pipeline
        """
        # Parse sub-folder inputs
        if inpt is not None:
            inpt = os.path.join(par_path, inpt)

        return inpt

    def parse_yaml(self, path: str) -> Dict:
        """
        Parses the input yaml file
        """
        with open(path, 'r') as yaml_file:
            args = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return args

    def _get_command_line_inputs(self) -> argparse.Namespace:

        self.parser = argparse.ArgumentParser(conflict_handler='resolve')

        # Input file is easiest way to pass arguments
        self.parser.add_argument(
            '--yaml', '-y',
            default=None,
            type=str,
            help='YAML file containing input parameters'
        )

        # Consider whether to add these
        self.parser.add_argument(
            '--output',
            type=str,
            default='output',
            help='Name of output directory. Default is output.'
        )
        self.parser.add_argument(
            '--overwrite',
            action='store_true',
            help='If set, will overwrite contents of output folder. Otherwise makes new folder.'
        )

        return self.parser.parse_arg()
