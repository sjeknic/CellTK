import os
import sys
import argparse
import yaml
import warnings
from multiprocessing import Pool
from typing import Dict, List, Collection, Tuple
from copy import deepcopy
from glob import glob

import numpy as np
import tifffile as tiff
import imageio as iio

from cellst.operation import Operation
from cellst.utils._types import Image, Mask, Track, Arr, INPT_NAMES
from cellst.utils._types import Condition, Experiment
from cellst.utils.process_utils import condense_operations, extract_operations
from cellst.utils.utils import folder_name


class Pipeline():
    # TODO: need a better way to define where the path should be...
    file_location = os.path.dirname(os.path.realpath(__file__))

    __slots__ = ('_image_container', 'operations',
                 'parent_folder', 'output_folder',
                 'image_folder', 'mask_folder',
                 'track_folder', 'array_folder',
                 'operation_index', 'img_ext')

    def __init__(self,
                 parent_folder: str = None,
                 output_folder: str = None,
                 image_folder: str = None,
                 mask_folder: str = None,
                 track_folder: str = None,
                 array_folder: str = None,
                 file_extension: str = 'tif',
                 overwrite: bool = True
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

        TODO:
            - Should be able to parse args and load a yaml as well
        """
        # Define paths to find and save images
        self.img_ext = file_extension
        self._set_all_paths(parent_folder, output_folder, image_folder,
                            mask_folder, track_folder, array_folder)
        self._make_output_folder(overwrite)

        # Prepare for getting operations and images
        self._image_container = {}
        self.operations = []

    def __enter__(self) -> None:
        """
        """
        if not hasattr(self, '_image_container'):
            self._image_container = {}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        """
        self._image_container = None
        del self._image_container

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
                # Get images and pass to operation
                # try/except block is here to raise more helpful messages for user
                # TODO: Add logging of files and whether they were found
                try:
                    imgs, msks, trks, arrs = self._get_images_from_container(inpts)
                except KeyError as e:
                    raise KeyError(f'Failed to find all inputs for {oper} \n',
                                   e.__str__())

                # Save the results in the image container
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
            - Add logging
            - Allow for non-consecutive indices (how?)
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
            else:
                img_dtype = arr.dtype if img_dtype is None else img_dtype
                if arr.ndim != 3:
                    warnings.warn("Expected stack with 3 dimensions."
                                  f"Got {arr.ndim} for {name}", UserWarning)

                # Save files as tiff with consecutive idx
                for idx in range(arr.shape[0]):
                    name = os.path.join(save_folder, f"{otpt_type}{idx}.tiff")
                    tiff.imsave(name, arr[idx, ...].astype(img_dtype))

    def _input_output_handler(self):
        """
        Iterate through all the operations and figure out which images
        to save and which to remove.

        TODO:
            - Determine after which operation the stack is no longer needed
        """
        req_inputs = []
        req_outputs = []
        for o in self.operations:
            imgs = [tuple([i, Image.__name__]) for i in o.input_images]
            msks = [tuple([m, Mask.__name__]) for m in o.input_masks]
            trks = [tuple([t, Track.__name__]) for t in o.input_tracks]
            arrs = [tuple([a, Arr.__name__]) for a in o.input_arrays]

            req_inputs.append([imgs, msks, trks, arrs])
            req_outputs.append(o.output_id)

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
            ext = self.img_ext in im
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
            - Not sure if it is finding image metadata. Test with images with metadata.
            - img_dtype should not scale an image up - an 8bit image should stay 8bit
        """
        for idx, name in enumerate(INPT_NAMES):
            fol = getattr(self, f'{name}_folder')

            # Get unique list of all inputs requested by operations
            all_requested = [sl for l in [i[idx] for i in inputs] for sl in l]
            to_load = list(set(all_requested))

            for key in to_load:
                pths = self._get_image_paths(fol, key[0])
                if len(pths) == 0:
                    # If no images are found in the path, check output_folder
                    fol = self.output_folder
                    pths = self._get_image_paths(fol, key[0])
                    # If still no images, check the listed outputs
                    if len(pths) == 0 and key not in outputs:
                        # TODO: The order matters. Should raise error if it is made
                        #       after it is needed.
                        raise ValueError(f'Data {key} cannot be found and is '
                                         'not listed as an output.')

                    # Don't try to load images if none found
                    continue


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

                # TODO: Why is this raising an error? Trying to load images that don't exist
                container[key] = img_stack

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
            self.parent_folder = os.path.abspath(sys.argv[0])
        else:
            self.parent_folder = parent_folder

        # Output folder defaults to folder in parent_folder
        if output_folder is None:
            self.output_folder = os.path.join(self.parent_folder, 'outputs')
        else:
            self.output_folder = output_folder

        # All image folders default to output_folder or parent_folder
        self.image_folder = self.parent_folder if image_folder is None else image_folder
        self.mask_folder = self.output_folder if mask_folder is None else mask_folder
        self.track_folder = self.output_folder if track_folder is None else track_folder
        self.array_folder = self.output_folder if array_folder is None else array_folder

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

        # TODO: Add Logging file here

    @classmethod
    def _run_single_pipe(cls,
                         pipe: Dict,
                         oper: Dict,
                         cond_map: Dict = {}
                         ) -> Arr:
        """
        Creates a Pipeline object, adds operations, and runs.

        NOTE: This might be overkill, but don't want it to persist
              in memory.
        NOTE: Assumes it is okay to make changes in-place in oper,
              i.e. that it is a copy of the original dictionary
        """
        # If Extract has default name, set to name of folder or condition
        # NOTE: 'FUNCTIONS' key in dict has no effect with Extract
        fol = folder_name(pipe['parent_folder'])
        if 'extract' in oper and oper['extract']['condition'] == 'default':
            try:
                oper['extract']['condition'] = cond_map[fol]
            except KeyError:
                # Means condition map was bad
                warnings.warn(f'Could not find condition for {fol}. '
                              'Using default.', UserWarning)

        # Initialize pipeline and operations
        pipe = Pipeline(**pipe)
        opers = extract_operations(oper)

        with pipe:
            # Run pipeline, save results, and then del pipeline
            pipe.add_operations(opers)
            result = pipe.run()
            del pipe

        return result


class Orchestrator():
    file_location = os.path.dirname(os.path.realpath(__file__))

    __slots__ = ('pipelines', 'operations',
                 'parent_folder', 'output_folder',
                 'operation_index', 'img_ext', 'name',
                 'overwrite', 'save', 'condition_map')

    def __init__(self,
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
                 save_master_df: bool = True,
                 ) -> None:
        """
        Args:

        Returns:
            None

        TODO:
            - Should be able to parse args and load a yaml as well
        """
        # Save some values
        self.name = name
        self.img_ext = file_extension
        self.overwrite = overwrite
        self.save = save_master_df

        # Build the Pipelines and input/output paths
        self.pipelines = {}
        self._build_pipelines(parent_folder, output_folder, match_str,
                              image_folder, mask_folder, track_folder,
                              array_folder)
        self.condition_map = self._update_condition_map(condition_map)
        self._make_output_folder(self.overwrite)

        # Prepare for getting operations
        self.operations = []

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

        # Get a single dictionary that defines all operations
        operation_dict = condense_operations(self.operations)

        # Run with multiple cores or just a single core
        if n_cores > 1:
            results = self.run_multiple_pipelines(self.pipelines,
                                                  operation_dict,
                                                  n_cores=n_cores)
        else:
            results = []
            for fol, kwargs in self.pipelines.items():
                results.append(
                    Pipeline._run_single_pipe(kwargs,
                                              deepcopy(operation_dict),
                                              self.condition_map)
                )

        if self.save:
            # TODO: If results are saved, pass here, otherwise, use files
            self.build_experiment_file()

        return results

    def run_multiple_pipelines(self,
                               pipeline_dict: Dict,
                               operation_dict: Dict,
                               n_cores: int = 1
                               ) -> Collection[Arr]:
        """
        pipeline_dict holds path information for building ALL of the pipelines
            - key is subfolder, val is to be passed to Pipeline.__init__
        operation_dict holds the information for building ALL of the operations
            - key is operation

        TODO:
            - Not sure a copy of operation_dict has to be made here. It's going
              to get pickled, I think, so no changes that _run_single_pipe will
              apply to any other Pipeline. I think...
        """
        with Pool(n_cores) as pool:
            # Set up pool of workers
            multi = [pool.apply_async(Pipeline._run_single_pipe,
                                      args=(kwargs, deepcopy(operation_dict),
                                            self.condition_map))
                     for kwargs in pipeline_dict.values()]

            # Run pool and return results
            return [m.get().shape for m in multi]

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
                raise ValueError('Not all elements of operation are class Operation.')
        elif isinstance(operation, Operation):
            if index == -1:
                self.operations.append(operation)
            else:
                self.operations.insert(index, operation)
        else:
            raise ValueError(f'Expected type Operation, got {type(operation)}.')

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

    def _build_pipelines(self,
                         parent_folder: str,
                         output_folder: str,
                         match_str: str,
                         image_folder: str,
                         mask_folder: str,
                         track_folder: str,
                         array_folder: str
                         ) -> None:
        """
        """
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

        # Find all folders that will be needed for Pipelines
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

                # Add miscellaneous options
                self.pipelines[fol]['file_extension'] = self.img_ext
                self.pipelines[fol]['overwrite'] = self.overwrite

    def _update_condition_map(self, condition_map: Dict) -> Dict:
        """
        """
        if len(condition_map) > 0:
            return condition_map
        else:
            # Default is to use folder name (keys in pipeline)
            return {k: k for k in self.pipelines}

    def _make_output_folder(self,
                            overwrite: bool = True
                            ) -> None:
        """
        This will also be responsible for logging and passing the yaml
        TODO:
            - Add logging file
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
        """
        # Parse sub-folder inputs
        if inpt is not None:
            inpt = os.path.join(par_path, inpt)

        return inpt

    def parse_yaml(self, path: str) -> Dict:
        """
        Parses the input yaml file
        """
        with open(path, 'r') as yam:
            args = yaml.load(yam, Loader=yaml.FullLoader)
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
