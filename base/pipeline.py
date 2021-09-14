import os
import sys
import argparse
import yaml
import multiprocessing
from typing import Dict, List, Collection, Union, Generator, Tuple
from glob import glob
import datetime as dtdt

import numpy as np
import tifffile as tiff
from imageio import imread, mimread
import cv2

from base.operation import Operation
from base.utils import Image, Mask, Track
from base.utils import INPT, INPT_NAMES, INPT_IDX, INPT_NAME_IDX


class Pipeline():
    """
    This is the no longer outermost holder class. It will hold all of the Operation classes.
    """

    # Need a better way to define where the path should be...
    file_location = os.path.dirname(os.path.realpath(__file__))

    def __init__(self,
                 image_folder: str,
                 channels: Collection[str] = None,
                 mask_folder: str = None,
                 track_folder: str = None,
                 output_folder: str = None,
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
        """
        self.output_folder = image_folder if output_folder is None else output_folder
        self.operations = []

        self._make_output_folder(self.output_folder, overwrite)

        # Collect images and sort by channel
        self._image_container = {}
        for im_type, im in zip([Image, Mask, Track],
                               [image_folder, mask_folder, track_folder]
                               ):
            files = self.get_image_paths(im, channels)
            self._image_container = self._load_images_to_container(
                                                        self._image_container,
                                                        channels,
                                                        files,
                                                        kind=im_type)

    def get_image_paths(self,
                        folder: str,
                        channels: Collection[str] = None
                        ) -> Collection[str]:
        """
        TODO:
            - Add image selection based on regex
            - Should channels use regex or glob to match name
        """
        # Return empty list if no inputs
        if folder is None:
            return []

        # Find images and sort into list based on channel
        all_images = sorted(os.listdir(folder))
        if channels is not None:
            channel_images = [[a for a in all_images if c in a]
                              for c in channels]
        else:
            channels = ['all']
            channel_images = [all_images]

        # Get the full path to the images
        channel_images = [[os.path.join(folder, i) for i in c]
                          for c in channel_images]

        return channel_images

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
        """
        if isinstance(operation, Collection):
            if all([isinstance(o, Operation) for o in operation]):
                self.operations[index:index] = operation
            else:
                raise ValueError('Not all elements of operation are class Operation.')
        elif isinstance(operation, Operation):
            self.operations.insert(index, operation)
        else:
            raise ValueError(f'Expected type Operation, got {type(operation)}.')

        self.operation_index = {i: o for i, o in enumerate(self.operations)}

    def run(self) -> (Image, Mask, Track):
        """
        Runs all of the operations in self.operations on the images

        Args:

        Returns:
        """
        # for each operation
        # inputs = three lists of image stacks
        # outputs = tuple of expected name and output_type (as str for now)
        inputs, outputs = self._input_output_handler()

        for inpts, otpts, oper in zip(inputs, outputs, self.operations):
            # Get images and pass to operation
            imgs, msks, trks = self._get_images_from_container(inpts)
            oper_result = oper.run_operation(imgs, msks, trks)

            # Save the results in the image container
            self._image_container = self.update_image_container(
                                            self._image_container,
                                            oper_result,
                                            otpts)

            if oper.save:
                self.save_images(oper_result, otpts[0])

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

            '''TODO: Here the function needs to remove any stacks from the image handler
            that are no longer needed. That should have been defined by the input/output'''

        return container

    def _input_output_handler(self):
        """
        Iterate through all the operations and figure out which images
        to save and which to remove.
        If an input doesn't exist, try reloading the images.
        If an output that will be used as an input doesn't exist, raise error
        Also, remember the names that should get passed to each operation, based
        on what they request.
        """
        req_inputs = []
        for o in self.operations:
            imgs = [tuple([i, Image.__name__]) for i in o.input_images]
            msks = [tuple([m, Mask.__name__]) for m in o.input_masks]
            trks = [tuple([t, Track.__name__]) for t in o.input_tracks]

            req_inputs.append([imgs, msks, trks])

        req_outputs = [tuple([o.output, o._output_type.__name__]) for o in self.operations]
        # Need to do some thinking here about which inputs/outputs are used
        # where and what it means for them to be in or missing from one of the lists.

        return req_inputs, req_outputs

    def save_images(self,
                    stack: (Image, Mask, Track),
                    output_name: str,
                    img_dtype: type = np.int16) -> None:
        """
        TODO:
            - Test different iteration strategies for efficiency
            - Should not upscale images
            - Use type instead of str for output (if even needed)
            - Add logging
        """
        if stack.ndim != 3:
            raise ValueError('Expected image stack with 3 dimensions. '
                             f'Got {stack.ndim}')

        for idx in range(stack.shape[0]):
            name = os.path.join(self.output_folder, f'{output_name}{idx}.tiff')
            tiff.imsave(name, stack[idx, ...].astype(img_dtype))

    def _load_images_to_container(self,
                                  container: Dict[str, np.ndarray],
                                  channels: Collection[str],
                                  channel_images: Collection[str],
                                  kind: type = Image,
                                  img_dtype: type = np.int16,
                                  save_metadata: bool = False
                                  ) -> Dict[str, tuple]:
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
            - Does this really need container as an input, could just use self.
        """
        # If no image paths are passed, just return the container
        if len(channel_images) == 0:
            return container

        # Check that images are present for all channels
        assert len(channels) == len(channel_images), ('Same number of channels '
                                                      'and images must be '
                                                      'provided.')
        # Check that the mask is correct
        if kind not in INPT_NAMES and kind not in INPT:
            raise TypeError(f'Image kind must be one of {INPT}. Got{kind}.')
        elif not isinstance(kind, str):
            try:
                kind = kind.__name__
            except AttributeError:
                raise AttributeError('Did not understand desired image input.'
                                     f'Make sure to use custom types {INPT_NAMES}')

        # Load the images
        '''NOTE: Using mimread instead of imread to add the option of limiting
        the memory of the loaded image. However, mimread still only loads one
        image at a time, so it is unlikely to ever hit the memory limit.
        Could be worth tracking the memory and deleting large arrays or
        temporarily storing them in a file (if low-mem mode is true)
        Slightly faster than imread.'''
        for cname, cimgs in zip(channels, channel_images):
            # TODO: Is there a better imread function to use here?
            img_arrs = [np.asarray(mimread(c)[0]) for c in cimgs]

            '''NOTE: Due to how numpy arrays are stored in memory, writing to the last
            axis is faster than writing to the first. Therefore, the time axis is
            on the first axis'''
            img_stack = np.stack(img_arrs, axis=0).astype(img_dtype)
            # NOTE: won't work for 1D images. Does that matter?
            while img_stack.ndim < 3:
                img_stack = np.expand_dims(img_stack, axis=0)

            container[tuple([cname, kind])] = img_stack

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

    def _make_output_folder(self,
                            output_path: str,
                            overwrite: bool = True
                            ) -> None:
        """
        This will also be responsible for logging and passing the yaml
        """
        self.output_folder = os.path.join(self.file_location, output_path)
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
        return


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
