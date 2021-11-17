from typing import Collection
import warnings

import numpy as np

from cellst.operation import BaseExtract
from cellst.utils._types import Image, Mask, Track, Arr, CellArray
from cellst.utils.operation_utils import lineage_to_track


class Extract(BaseExtract):
    def extract_data_from_image(self,
                                images: Collection[Image],
                                masks: Collection[Mask] = [],
                                tracks: Collection[Track] = [],
                                channels: Collection[str] = [],
                                regions: Collection[str] = [],
                                lineages: Collection[np.ndarray] = [],
                                condition: str = None
                                ) -> Arr:
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
            - Add option to change padding value
        """
        # Check that all required inputs are there
        if len(tracks) == 0 and len(masks) == 0:
            raise ValueError('Missing masks and/or tracks.')
        if len(tracks) != 0:
            # Uses tracks if provided and skips masks
            tracks_to_use = tracks
        elif len(masks) != 0:
            tracks_to_use == masks
            # Check that sufficient lineages are provided
            if len(lineages) == 0:
                warnings.warn('Got mask but not lineage file. No cell division'
                              ' can be tracked.', UserWarning)
            elif len(masks) != len(lineages):
                # TODO: This could probably be a warning and pad lineages
                raise ValueError(f'Got {len(masks)} masks '
                                 f'and {len(lineages)} lineages.')
            else:
                tracks_to_use = [lineage_to_track(t, l)
                                 for t, l in zip(tracks_to_use, lineages)]

        # Confirm sizes of inputs match
        if len(images) != len(channels):
            warnings.warn(f'Got {len(images)} images '
                          f'and {len(channels)} channels.'
                          'Using default naming.', UserWarning)
            channels = [f'image{n}' for n in range(len(images))]
        if len(tracks_to_use) != len(regions):
            warnings.warn(f'Got {len(tracks_to_use)} tracks '
                          f'and {len(regions)} regions.'
                          'Using default naming.', UserWarning)
            regions = [f'region{n}' for n in range(len(tracks_to_use))]

        # TODO: Add handling of extra_properties
        # Label must always be the first metric for easy indexing of cells
        metrics = self._metrics
        if 'label' not in metrics:
            metrics.insert(0, 'label')
        elif metrics[0] != 'label':
            metrics.remove('label')
            metrics.insert(0, 'label')

        # Get unique cell indexes and the number of frames
        cells = np.unique(np.concatenate([t[t > 0] for t in tracks]))
        cell_index = {int(a): i for i, a in enumerate(cells)}
        frames = range(max([i.shape[0] for i in images]))

        # Initialize data structure
        data = CellArray(regions, channels, metrics, cells, frames,
                         name=condition)

        # Extract data for all channels and regions individually
        for c_idx, cnl in enumerate(channels):
            for r_idx, rgn in enumerate(regions):
                cnl_rgn_data = self._extract_data_with_track(images[c_idx],
                                                             tracks[r_idx],
                                                             metrics,
                                                             cell_index)
            data[rgn, cnl, :, :, :] = cnl_rgn_data

        # Does not need to return type to Extract.run_operation
        return data
