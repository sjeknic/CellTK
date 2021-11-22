import os
import sys
import argparse
import yaml
import multiprocessing
import warnings
from typing import Dict, List, Collection, Union, Generator, Tuple
from glob import glob
import datetime as dtdt

import numpy as np
import tifffile as tiff
import imageio as iio

from cellst.operation import Operation
from cellst.utils._types import Image, Mask, Track, Arr
from cellst.utils._types import INPT, INPT_NAMES, INPT_IDX, INPT_NAME_IDX


class Pipeline():
    """
    This is the no longer outermost holder class. It will hold all of the Operation classes.
    """

    # TODO: need a better way to define where the path should be...
    file_location = os.path.dirname(os.path.realpath(__file__))

    __slots__ = ('_image_container', 'operations',
                 'parent_folder', 'output_folder',
                 'image_folder', 'mask_folder',
                 'track_folder', 'array_folder',
                 'operation_index')

    def __init__(self,
                 parent_folder: str = None,
                 output_folder: str = None,
                 image_folder: str = None,
                 mask_folder: str = None,
                 track_folder: str = None,
                 array_folder: str = None,
                 input_yaml: str = None,
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
            - Look at context management: https://stackoverflow.com/a/34325346
                Here and/or in Orchestrator to ensure image_containers are deleted
        """
        # Define paths to find and save images
        self._set_all_paths(parent_folder, output_folder, image_folder,
                            mask_folder, track_folder, array_folder)
        self._make_output_folder(overwrite)

        # Prepare for getting operations and images
        self._image_container = {}
        self.operations = []

    def add_operations(self,
                       operation: Collection[Operation],
                       index: int = -1,
                       save: (bool, Collection[bool]) = False
                       ) -> None:
        """
        Adds Operations to the Pipeline and recalculates the index

        Args:

        Returns:

        TODO:
            - Add option for saving individual results of the output
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
            oper_result = oper.run_operation(imgs, msks, trks, arrs)
            self._image_container = self.update_image_container(
                                            self._image_container,
                                            oper_result,
                                            otpts)

            # Write to disk if needed
            if oper.save:
                self.save_images(oper.save_arrays, oper._output_type.__name__)

        return oper_result

    def update_image_container(self,
                               container: Dict[str, np.ndarray],
                               array: (Image, Mask, Track),
                               key: Tuple[str],
                               ) -> None:
        """
        TODO:
            - How to handle if array is not an image stack (say df or something.)
                Probably just still save it all the same I think.
        """
        if key not in container:
            container[key] = array
        else:
            # TODO: Should there be an option to not overwrite?
            container[key] = array

            '''TODO: Here the function needs to remove any stacks from the image handler
            that are no longer needed. That should have been defined by the input/output'''

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
            - If an output that will be used as an input doesn't exist, raise error
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

        # Need to do some thinking here about which inputs/outputs are used
        # where and what it means for them to be in or missing from one of the lists.

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

        # TODO: Tiff should not be hardcoded
        # Find images and sort into list
        im_names = [os.path.join(folder, im)
                    for im in sorted(os.listdir(folder))
                    if match_str in im if 'tif' in im
                    and os.path.isfile(os.path.join(folder, im))]

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
                            if 'tif' in im
                            if os.path.isfile(os.path.join(folder, im))]
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
            - Potentially move to a utils class so it can be inherited by other classes
              If so, should no longer be private
            - Not sure if it is finding image metadata. Test with images with metadata.
            - img_dtype should not scale an image up - an 8bit image should stay 8bit
        """
        for idx, name in enumerate(INPT_NAMES):
            fol = getattr(self, f'{name}_folder')

            # Get unique list of all inputs requested by operations
            all_requested = [sl for l in [i[idx] for i in inputs] for sl in l]
            to_load = list(set(all_requested))

            for key in to_load:
                print(key)
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

                # Load the images
                '''NOTE: Using mimread instead of imread to add the option of limiting
                the memory of the loaded image. However, mimread still only loads one
                file at a time, so it is unlikely to ever hit the memory limit.
                Could be worth tracking the memory and deleting large arrays or
                temporarily storing them in a file (if low-mem mode is true)
                Slightly faster than imread.'''
                # Pre-allocate numpy array for speed
                for n, p in enumerate(pths):
                    print(p)
                    # TODO: Is there a better imread function to use here?
                    # TODO: Include check for image format
                    img = iio.mimread(p)[0]

                    # TODO: This would be faster as a try/except block
                    if n == 0:
                        # TODO: Is there a neater way to do this?
                        '''NOTE: Due to how numpy arrays are stored in memory, writing to the last
                        axis is faster than writing to the first. Therefore, the time axis is
                        on the first axis'''
                        img_stack = np.empty(tuple([len(pths), img.shape[0], img.shape[1]]),
                                             dtype=img.dtype)

                    img_stack[n, ...] = img

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


class Orchestrator():
    """
    Not sure if I will include this yet. The idea is that this would be the interface with fireworks
    or other ambiguous job scheduler. Basically it just needs to create a separate Pipeline for
    each folder it is given.

    This would also be used for running Pipelines on multiple local cores.
    """
    file_location = os.path.dirname(os.path.realpath(__file__))

    def __init__(self):
        # Get the user inputs (path to yaml for now...)
        self._get_command_line_inputs()

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
