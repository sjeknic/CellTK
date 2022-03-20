import warnings
from typing import Collection, Tuple, Union


import numpy as np

from celltk.core.operation import BaseExtractor
from celltk.utils.utils import ImageHelper
from celltk.utils._types import Image, Mask, Track, Arr
from celltk.core.arrays import ConditionArray
from celltk.utils.operation_utils import lineage_to_track, parents_from_track


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
                                time: Union[float, np.ndarray] = None,
                                condition: str = 'default',
                                position_id: int = None,
                                min_trace_length: int = 0,
                                skip_frames: Tuple[int] = tuple([]),
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
            tracks_to_use = list(tracks)
        if len(masks) != 0:
            # Check that sufficient lineages are provided
            if len(lineages) == 0:
                warnings.warn('Got mask but not lineage file.', UserWarning)
                tracks_to_use.extend(list(masks))
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
        self.skip_frames = skip_frames
        cells = np.unique(np.concatenate([t[t > 0] for t in tracks_to_use]))
        cell_index = {int(a): i for i, a in enumerate(cells)}
        frames = max([i.shape[0] for i in images])
        if self.skip_frames: frames += len(self.skip_frames)
        frames = range(frames)

        # Initialize data structure
        array = ConditionArray(regions, channels, all_measures, cells, frames,
                               name=condition, pos_id=position_id)

        # Check to see if all axes have something
        if any([not a for a in array.shape]):
            missing = [k for k, a in zip(array.coordinates, array.shape)
                       if not a]
            # If axes are missing, we skip everything and save nothing.
            warnings.warn(f'The following dimensions are missing: {missing}')
        if time: array.set_time(time)

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

        # Remove short traces
        self.logger.info(f'Removing cells with traces < {min_trace_length} frames.')
        self.logger.info(f'Current array size: {array.shape}')
        mask = array.remove_short_traces(min_trace_length)
        array.filter_cells(mask, delete=True)
        self.logger.info(f'Post-filter array size: {array.shape}')

        # Check for calculated metrics to add and filters
        # TODO: Does it make a difference before or after parent??
        self._calculate_derived_metrics(array)
        self._apply_filters(array)

        return array

    def add_extra_metric(self, name: str, func: Callable = None) -> None:
        """
        Allows for adding custom metrics. If function is none, value will just
        be nan.

        TODO: Callable function won't be saveable in YAML files
        """
        if not func:
            if name in self._possible_metrics:
                self._metrics.append(name)
            else:
                try:
                    func = getattr(metric_utils, name)
                except AttributeError:
                    # Function not implemented by me
                    func = RandomNameProperty()
        else:
            assert isinstance(func, Callable)

        self._props_to_add[name] = func

    def add_derived_metric(self,
                           metric_name: str,
                           keys: Collection[Tuple[str]],
                           func: str = 'sum',
                           inverse: bool = False,
                           propagate: (str, bool) = False,
                           frame_rng: Union[int, Tuple[int]] = None,
                           *args, **kwargs
                           ) -> None:
        """
        Calculates additional metrics based on information already in array
        func can be any numpy function
        propagate can be bool, or the name of dimension to propagate to

        TODO: Add possiblity for custom Callable function
        TODO: Peaks could probably be passed here???
        TODO: So could segmentation of peaks???
        """
        # Check the inputs now before calculation
        # Assert that keys include channel, region, and metric
        peak_metrics = ('predict_peaks', 'active_cells',
                        'cumulative_active', 'active')
        for key in keys:
            assert len(key) == 3
        if not hasattr(np, func) and not hasattr(metric_utils, func):
            if metric_name not in peak_metrics:
                raise ValueError('Metric must be numpy func '
                                 'or in metric_utils.')

        # Save to calculated metrics to get added after extract is done
        # TODO: Make a dictionary
        self._derived_metrics[metric_name] = tuple([func, keys, inverse, propagate,
                                                    frame_rng, args, kwargs])

        # Fill in the metric with just nan for now
        if metric_name in peak_metrics:
            if metric_name == 'predict_peaks':
                self._props_to_add['slope_prob'] = RandomNameProperty()
                self._props_to_add['plateau_prob'] = RandomNameProperty()
                self._props_to_add['peaks'] = RandomNameProperty()
            else:
                self._props_to_add['active'] = RandomNameProperty()
                self._props_to_add['cumulative_active'] = RandomNameProperty()
        else:
            self._props_to_add[metric_name] = RandomNameProperty()

        self.logger.info(f'Added derived metric {metric_name}')

    def add_filter(self,
                   filter_name: str,
                   metric: str,
                   region: Union[str, int] = 0,
                   channel: Union[str, int] = 0,
                   frame_rng: Union[int, Tuple[int]] = None,
                   *args, **kwargs
                   ) -> None:
        """
        TODO: Add ability to pass Callable, has to be done after Extract now
        """
        assert hasattr(filter_utils, filter_name), f'{filter_name} not found.'
        added_metrics = (self._extra_properties
                         + self._metrics
                         + list(self._derived_metrics.keys()))
        assert metric in added_metrics, f'Metric {metric} not found'

        # TODO: Make a dictionary
        self._filters.append(dict(filter_name=filter_name, metric=metric, region=region,
                                  channel=channel, frame_rng=frame_rng,
                                  args=args, kwargs=kwargs))

    def set_metric_list(self, metrics: Collection[str]) -> None:
        """Sets the list of metrics to get. For a possible list, see
        skimage.regionprops or Extractor._possible_metrics.

        :param metrics: List of metrics to measure from images

        :return: None

        NOTE:
            - CellTK can only use the scalar metrics in regionprops.
        """
        # Check that skimage can handle the given metrics
        allowed = [m for m in metrics if m in self._possible_metrics]
        not_allowed = [m for m in metrics if m not in self._possible_metrics]

        self._metrics = allowed

        # Raise warning for the rest
        if not_allowed:
            warnings.warn(f'Metrics {[not_allowed]} are not supported. Use '
                          'CellArray.add_extra_metric to add custom metrics.')
