from typing import Collection, Tuple, Dict, Callable

import numpy as np
import skimage.measure as meas

from cellst.utils.utils import image_helper, Image, Mask, Track, Arr, INPT_NAME_IDX
from cellst.custom_array import CustomArray


class Operation(object):
    """
    This is the base class for the operations (segmentation, tracking, etc.)

    TODO:
        - Implement __slots__
    """

    def __init__(self,
                 output: str,
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 **kwargs
                 ) -> None:
        """
        by default only the last output can be saved (should this change?)

        TODO:
            - Add more name options, like a specific output folder name
            - Outputs are now a required arg for the base class.
            - Add name or other easy identifier for error messages
        """
        self.save = save
        self.output = output

        # These are used to track what the operation has been asked to do
        self.functions = []
        self.func_index = {}

        # Will be overwritten by inheriting class if they are used
        # Otherwise, these defaults can be used to know if they haven't
        self.input_images = []
        self.input_masks = []
        self.input_tracks = []
        self.input_arrays = []

        if _output_id is not None:
            self.output_id = _output_id
        else:
            output_type = self._output_type.__name__
            self.output_id = tuple([output, output_type])

    def __setattr__(self, name, value) -> None:
        '''TODO: Not needed here, but the idea behind a custom __setattr__
               class is that the inheriting Operation can decide if the function
               meets the requirements.'''
        super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        Returns printable version of the functions and args in Operation

        TODO:
            - Return function name instead of decorator name
        """
        string = str(super().__str__())

        for k, v in self.func_index.items():
            string += (f'\nIndex {k}: \n'
                       f'Function: {v[0]} \n'
                       f'   args: {v[1]} \n'
                       f'   kwargs: {v[2]}')
        return string

    def add_function_to_operation(self,
                                  func: str,
                                  output_type: type = None,
                                  index: int = -1,
                                  *args,
                                  **kwargs
                                  ) -> None:
        """
        args and kwargs should be passed to the function.

        TODO:
            - Add option for func to be a Callable
            - Is func_index needed at all?
        """
        if not hasattr(self, func):
            raise NameError(f"Function {func} not found in {self}.")
        else:
            func = getattr(self, func)
            self.functions.append(tuple([func, output_type, args, kwargs]))

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

        TODO:
            - np array could be preallocated for the function
        """
        # Default is to return the input if no function is run
        inputs = [images, masks, tracks, arrays]
        result = inputs[INPT_NAME_IDX[self._output_type.__name__]]

        for (func, expec_type, args, kwargs) in self.functions:
            # TODO: Only gets output type from decorator or if
            #       function explicitly returns it
            output_type, result = func(*inputs, *args, **kwargs)

            # The user-defined expected type will overwrite output_type
            output_type = expec_type if expec_type is not None else output_type

            # Pass the result to the next function
            # TODO: This will currently raise a KeyError if it gets an unexpected type
            if isinstance(result, np.ndarray):
                inputs[INPT_NAME_IDX[output_type.__name__]] = [result]
            else:
                inputs[INPT_NAME_IDX[output_type.__name__]] = result

        return result


class Processing(Operation):
    _input_type = (Image,)
    _output_type = Image

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
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


class Segment(Operation):
    _input_type = (Image,)
    _output_type = Mask

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 output: str = 'mask',
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        self.output = output

    # TODO: Should these methods actually be static? What's the benefit?
    @staticmethod
    @image_helper
    def constant_thres(image: Image,
                       THRES=1000,
                       NEG=False
                       ) -> Mask:
        if NEG:
            return meas.label(image < THRES).astype(np.int16)
        return meas.label(image > THRES).astype(np.int16)

    @image_helper
    def unet_predict(self,
                     image: Image,
                     weight_path: str,
                     roi: (int, str) = 2,
                     batch: int = None,
                     classes: int = 3,
                     ) -> Image:
        """
        NOTE: If we had mulitple colors, then image would be 4D here. The Pipeline isn't
        set up for that now, so for now the channels is just assumed to be 1.

        roi - the prediction values are returned only for the roi
        batch - number of frames passed to model. None is all of them.
        classes - number of output categories from the model (has to match weights)
        """
        # probably don't need as many options here
        _roi_dict = {'background': 0, 'bg': 0, 'edge': 1,
                     'interior': 2, 'nuc': 2, 'cyto': 2}
        if isinstance(roi, str):
            try:
                roi = _roi_dict[roi]
            except KeyError:
                raise ValueError(f'Did not understand region of interest {roi}.')

        # Only import tensorflow and Keras if needed
        from base.unet_utils import unet_model

        if not hasattr(self, 'model'):
            '''NOTE: If we had mulitple colors, then image would be 4D here.
            The Pipeline isn't set up for that now, so for now the channels
            is just assumed to be 1.'''
            channels = 1
            dims = (image.shape[1], image.shape[2], channels)

            self.model = unet_model.UNetModel(dimensions=dims,
                                              weight_path=weight_path,
                                              model='unet')

        # Pre-allocate output memory
        # TODO: Incorporate the batch here.
        if batch is None:
            output = self.model.predict(image[:, :, :], roi=roi)
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a, roi=roi)
                                     for a in arrs], axis=0)

        # TODO: dtype can probably be downsampled from float32 before returning
        return output


class Track(Operation):
    _input_type = (Image, Mask)
    _output_type = Track

    def __init__(self,
                 input_images: Collection[str] = [],
                 input_masks: Collection[str] = [],
                 output: str = 'track',
                 save: bool = False,
                 track_file: bool = True,
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


class Evaluate(Operation):
    _input_type = (Arr,)
    _output_type = Arr
    pass


class Extract(Operation):
    _input_type = (Image, Mask, Track)
    _output_type = Arr
    # Label will always be first, even for user supplied metrics
    """TODO: This is important. xr.DataArray can only handle a single datatype.
    This is similar to CellTK/covertrace. However, it prevents the use of some interesting
    properties, such as coords. An approach to fix this is to use an xr.DataSet to hold multiple
    xr.DataArrays of multiple types. However, I am concerned that this will
    greatly slow down the indexing, and it also complicates returning multiple
    values. For now, I'm just going to focus on metrics that can be stored as scalars.
    TODO: Additionally, some of the metrics that can be stored as scalars are returned
    as multiple metrics (i.e. bbox = bbox-0, bbox-1, bbox-2, bbox-3). These need to be
    automatically handled. See scipy.regionprops and scipy.regionprops_table for
    more information on which properites."""
    _metrics = ['label', 'area', 'convex_area',
                # 'equivalent_diameter_area',
                # 'centroid_weighted',  # centroid weighted by intensity
                'mean_intensity',
                # 'intensity_mean', 'intensity_min',  # need to add median intensity
                'orientation', 'perimeter', 'solidity',
                # 'bbox', 'centroid', 'coords'  # require multiple scalars
                ]
    _extra_properties = []

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 input_masks: Collection[str] = [],
                 input_tracks: Collection[str] = [],
                 channels: Collection[str] = [],
                 regions: Collection[str] = [],
                 condition: str = None,
                 output: str = 'data_frame',
                 save: bool = False,
                 _output_id: Tuple[str] = None
                 ) -> None:
        """
        channel and region _map should be the names that will get saved in the final df
        with the images and masks they correspond to.
        TODO:
            - Clean up this whole function when it's working
            - metrics needs to always start with label
            - If Mask is given, but not track, look for tracking file
                - Raise warning that parent daughter connections will fail if missing
            - Condition should get passed to this function. Pipeline likely cannot because
              the same operation will be used for multiple Pipelines. But Orchestrator should
              be able to.
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
            self.channels = input_images
        else:
            self.channels = channels

        # TODO: This should look for both masks and tracks
        if len(regions) == 0:
            self.regions = input_masks
        else:
            self.regions = regions

        # Only one function can be used for the Extract module
        # TODO: Overwrite add_function to raise an error, user was
        #       likely trying to add extra_properties instead
        self.functions = [tuple([self.extract_data_from_image, Arr, [], {}])]
        self.func_index = {i: f for i, f in enumerate(self.functions)}

    def extract_data_from_image(self,
                                images: Collection[Image],
                                masks: Collection[Mask],
                                tracks: Collection[Track],
                                array: Collection[Arr] = [],
                                *args) -> Arr:
        """
        ax 0 - cell locations (nuc, cyto, population, etc.)
        ax 1 - channels (TRITC, FITC, etc.)
        ax 2 - metrics (median_int, etc.)
        ax 3 - cells
        ax 4 - frames

        TODO:
            - Allow an option for caching or not in regionprops
        """
        '''TODO: Track should return a mask that starts at 0 as background,
        and increments by one for the number of cells. This simplifies finding
        the cells in the regionprops_table.'''
        '''TODO: Track should also link the parent and daughter traces? Actually
        no, thats not possible until here. It's only after regionprops that I can
        figure out the traces? So then after building the table, I can do the linking?
        But then if I remove daughters they will no longer be indexed properly.
        I guess with the tracking mask and/or the tracking file, I will know which
        daughters arrived when. Then for all the frames preceding, just fill in the
        information from the parent. Which will already be in data, so it can just
        be copied?'''

        # TODO: Add handling of extra_properties
        # Label must always be the first metric for easy indexing of cells
        metrics = self._metrics
        if 'label' not in metrics:
            metrics.insert(0, 'label')
        else:
            if metrics[0] != 'label':
                while True:
                    try:
                        metrics.remove('label')
                    except ValueError:
                        break

                metrics.insert(0, 'label')

        cells = np.unique(np.concatenate([np.unique(m) for m in masks]))
        cells_index = {int(a): i for i, a in enumerate(cells)}
        frames = range(max([i.shape[0] for i in images]))

        # Initialize data structure
        data = CustomArray(self.regions, self.channels, metrics, cells, frames)

        # Iterate through all channels and masks
        for c_idx, cnl in enumerate(self.channels):
            for r_idx, rgn in enumerate(self.regions):

                # TODO: Remember to include Tracks as a possible input
                # Extract data using scipy
                rp = [meas.regionprops_table(masks[r_idx][i], images[c_idx][i],
                                             properties=metrics, cache=True)
                      for i in range(images[c_idx].shape[0])]

                # This is used for padding empty values with np.nan
                all_nans = np.empty((len(frames), len(metrics), len(cells)))
                all_nans[:] = np.nan

                for frame in frames:
                    # TODO: Probably need a check for all scalars, either here or elsewhere
                    # TODO: Parent-daughter linking has to happen somewhere around here.
                    frame_data = np.row_stack(tuple(rp[frame].values()))

                    # Label is in the first position
                    for n, lab in enumerate(frame_data[0, :]):
                        all_nans[frame, :, cells_index[int(lab)]] = frame_data[:, n]

            # Don't need to explicitly set the last indices
            # Need to move the frames from the first axis to the last
            # And is this actually faster than just writing to the desired location to start.
            data[rgn, cnl, :, :, :] = np.moveaxis(all_nans, 0, -1)

        # Needs to return output_type to be consistent
        # TODO: This should be corrected
        return Arr, data
