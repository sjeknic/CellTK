from typing import Collection
import warnings

import numpy as np

from cellst.core.operation import BaseExtractor
from cellst.utils.utils import ImageHelper
from cellst.utils._types import Image, Mask, Track, Arr
from cellst.core.arrays import ConditionArray
from cellst.utils.operation_utils import lineage_to_track, parents_from_track


class Extractor(BaseExtractor):
    _metrics = ['label', 'area', 'convex_area', 'filled_area', 'bbox',
                'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                'minor_axis_length', 'major_axis_length',
                'orientation', 'perimeter', 'solidity']
    _extra_properties = ['division_frame', 'parent_id', 'total_intensity',
                         'median_intensity']

    @ImageHelper(by_frame=False, as_tuple=True)
    def extract_data_from_image(self,
                                images: Image,
                                masks: Mask = [],
                                tracks: Track = [],
                                channels: Collection[str] = [],
                                regions: Collection[str] = [],
                                lineages: Collection[np.ndarray] = [],
                                condition: str = 'default',
                                position_id: int = None,
                                min_trace_length: int = 0,
                                remove_parent: bool = True,
                                parent_track: int = 0
                                ) -> Arr:
        """
        ax 0 - cell locations (nuc, cyto, population, etc.)
        ax 1 - channels (TRITC, FITC, etc.)
        ax 2 - metrics (median_int, etc.)
        ax 3 - cells
        ax 4 - frames

        Args:
            - image, masks, tracks = self-explanatory
            - channels - names associated with images
            - regions - names associated with tracks
            - lineages - if masks are provided
            - condition - name of dataframe
            - remove_parent - if true, use a track to connect par_daught
                              and remove parents
            - parent_track - if remove_parent, track to use for lineage info

        TODO:
            - Allow an option for caching or not in regionprops
            - Allow input of tracking file
            - Add option to change padding value
        """
        # Check that all required inputs are there
        if len(tracks) == 0 and len(masks) == 0:
            raise ValueError('Missing masks and/or tracks.')

        # Collect the tracks to use
        tracks_to_use = []
        if len(tracks) != 0:
            # Uses tracks first if provided
            tracks_to_use = tracks
        if len(masks) != 0:
            # Check that sufficient lineages are provided
            if len(lineages) == 0:
                warnings.warn('Got mask but not lineage file. No cell division'
                              ' can be tracked.', UserWarning)
                tracks_to_use.extend(masks)
            elif len(masks) != len(lineages):
                # TODO: This could probably be a warning and pad lineages
                raise ValueError(f'Got {len(masks)} masks '
                                 f'and {len(lineages)} lineages.')
            else:
                tracks_to_use.extend([lineage_to_track(t, l)
                                     for t, l in zip(tracks_to_use, lineages)])

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

        # Get all of the metrics and functions that will be run
        metrics = self._metrics
        extra_names = list(self._props_to_add.keys())
        extra_funcs = list(self._props_to_add.values())
        all_measures = self._correct_metric_dim(metrics + extra_names)

        # Label must always be the first metric for easy indexing of cells
        if 'label' not in all_measures:
            all_measures.insert(0, 'label')
        elif metrics[0] != 'label':
            all_measures.remove('label')
            all_measures.insert(0, 'label')

        # Get unique cell indexes and the number of frames
        cells = np.unique(np.concatenate([t[t > 0] for t in tracks_to_use]))
        cell_index = {int(a): i for i, a in enumerate(cells)}
        frames = range(max([i.shape[0] for i in images]))

        # Initialize data structure
        array = ConditionArray(regions, channels, all_measures, cells, frames,
                               name=condition, pos_id=position_id)

        # Extract data for all channels and regions individually
        for c_idx, cnl in enumerate(channels):
            for r_idx, rgn in enumerate(regions):
                cnl_rgn_data = self._extract_data_with_track(
                    images[c_idx],
                    tracks_to_use[r_idx],
                    metrics,
                    extra_funcs,
                    cell_index
                )
            array[rgn, cnl, :, :, :] = cnl_rgn_data

        if remove_parent:
            # Get parent information from a single track
            parent_track = tracks_to_use[parent_track]
            parent_lookup = parents_from_track(parent_track)

            # Build parent mask
            mask = array.remove_parents(parent_lookup, cell_index)

            # Remove cells
            array.filter_cells(mask, delete=True)

        # Check for calculated metrics to add
        # TODO: Does it make a difference before or after parent??
        self._calculate_derived_metrics(array)

        # Remove short traces
        mask = array.remove_short_traces(min_trace_length)
        array.filter_cells(mask, delete=True)

        return array
