from typing import Collection, Tuple
import warnings

import numpy as np
from skimage.measure import regionprops_table

from cellst.operation import BaseExtract
from cellst.utils.utils import ImageHelper
from cellst.utils._types import Image, Mask, Track, Arr, CellArray
from cellst.utils.operation_utils import lineage_to_track


class Extract(BaseExtract):
    @ImageHelper(by_frame=False)
    def extract_data_from_image(self,
                                images: Collection[Image],
                                masks: Collection[Mask] = [],
                                tracks: Collection[Track] = [],
                                channels: Collection[str] = [],
                                regions: Collection[str] = [],
                                lineages: Collection[np.ndarray] = [],
                                condition: str = None,
                                *args) -> Arr:
        """
        ax 0 - cell locations (nuc, cyto, population, etc.)
        ax 1 - channels (TRITC, FITC, etc.)
        ax 2 - metrics (median_int, etc.)
        ax 3 - cells
        ax 4 - frames

        NOTE: Extract takes in all inputs, so no need for decorator
        NOTE: Inputs are optional so that __call__ can use either mask
              or track

        TODO:
            - Allow an option for caching or not in regionprops
            - Allow input of tracking file
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

        # Check that all required inputs are there
        if len(tracks) == 0 and len(masks) == 0:
            raise ValueError(f'Missing masks and/or tracks.')
        if len(tracks) != 0:
            tracks_to_use = tracks
        elif len(masks) != 0:
            if len(lineages) == 0:
                warnings.warn('Got mask but not lineage file. No cell division'
                              ' can be tracked', UserWarning)
                tracks_to_use = masks
            else:
                if len(masks) != len(tracks):
                    raise ValueError(f'Got {len(masks)} masks '
                                     f'and {len(lineages)} lineages.')
                tracks_to_use = [lineage_to_track(m, l)
                                 for m, l in zip(masks, lineages)]

        # Confirm sizes of inputs match
        if len(images) != len(channels):
            raise ValueError(f'Got {len(images)} images and {len(channels)} channels.')
        if len(tracks_to_use) != len(regions):
            raise ValueError(f'Got {len(tracks)} tracks and {len(regions)} regions.')

        # TODO: Add handling of extra_properties
        # Label must always be the first metric for easy indexing of cells
        metrics = self._metrics
        if 'label' not in metrics:
            metrics.insert(0, 'label')
        else:
            if metrics[0] != 'label':
                try:
                    metrics.remove('label')
                except ValueError:
                    pass
                metrics.insert(0, 'label')

        # Get unique cell indexes and the number of frames
        cells = np.unique(np.concatenate([np.unique(m) for m in masks]))
        cells_index = {int(a): i for i, a in enumerate(cells)}
        frames = range(max([i.shape[0] for i in images]))

        # Initialize data structure
        data = CellArray(regions, channels, metrics, cells, frames,
                         name=condition)

        # Iterate through all channels and masks
        for c_idx, cnl in enumerate(channels):
            for r_idx, rgn in enumerate(regions):

                # TODO: Remember to include Tracks as a possible input
                # Extract data using scipy
                rp = [regionprops_table(tracks_to_use[r_idx][i], images[c_idx][i],
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
            data[rgn, cnl, :, :, :] = np.moveaxis(all_nans, 0, -1)

        # Needs to return output_type to be consistent
        return Arr, data
