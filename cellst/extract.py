from typing import Collection, Tuple

import numpy as np
import skimage.measure as meas

from cellst.operation import Operation
from cellst.utils.utils import Image, Mask, Track, Arr
from cellst.custom_array import CustomArray


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