import warnings
from typing import Collection, Tuple, Union, Callable

import numpy as np

from celltk.core.operation import BaseExtract
from celltk.utils.utils import ImageHelper
from celltk.utils._types import Image, Mask, Array, RandomNameProperty
from celltk.core.arrays import ConditionArray
from celltk.utils.operation_utils import lineage_to_track, parents_from_track
import celltk.utils.metric_utils as metric_utils


class Extract(BaseExtract):
    _metrics = ['label', 'area', 'convex_area', 'filled_area', 'bbox',
                'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                'minor_axis_length', 'major_axis_length',
                'orientation', 'perimeter', 'solidity']
    _extra_properties = ['division_frame', 'parent_id', 'total_intensity',
                         'median_intensity']

    @ImageHelper(by_frame=False, as_tuple=True)
    def extract_data_from_image(self,
                                images: Image,
                                masks: Mask,
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
                                ) -> Array:
        """Extracts data from stacks of images and constructs a ConditionArray.

        :param images: Images to extract data from.
        :param masks: Masks to segment images with.
        :param channels: Names of channels corresponding to
        :param regions: Names of segmented regions corresponding,
            tracks and masks, in that order.
        :param lineages: Lineage files corresponding to masks if provided.
        :param time: If int or float, designates time between frames.
            If array, marks the frame time points.
        :param condition: Name of the condition
        :param position_id: Unique identifier if multiple ConditionArrays
            will share the same condition
        :param min_trace_length: All cells with shorter traces will
            be deleted from the final array.
        :param skip_frames: Use to specify frames to be skipped. If provided
            to Pipeline, does not need to be provided again, but must match.
        :param remove_parent: If true, parents of cells are not kept in
            the final ConditionArray.
        :param parent_track: If multiple tracks are provided, designates the
            one to use for lineage tracking

        :return: ConditionArray with data from the images.
        :rtype: ConditionArray

        The following axes dimensions are used:
        ax 0 - cell locations (nuc, cyto, population, etc.)
        ax 1 - channels (TRITC, FITC, etc.)
        ax 2 - metrics (median_int, etc.)
        ax 3 - cells
        ax 4 - frames

        TODO:
            - Allow an option for caching or not in regionprops
            - Allow input of tracking file
            - Add option to change padding value
        """
        # Determine which masks contain tracking information:
        tracks = []
        _masks = []
        for m in masks:
            if (m < 0).any():
                tracks.append(m)
            else:
                _masks.append(m)
        masks = _masks

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
        Add custom metrics or metrics from regionprops to array.
        If function is none, value will just be nan.

        :param name: key for the metric in the ConditionArray
        :param func: If str, name of the metric from skiamge.regionprops.
            if Callable, function that calculates the metric. Cannot
            be used if Operation must be saved as YAML before running.

        :rtype: None

        TODO:
            - Callable function won't be saveable in YAML files
            - Add link to regionprops to docstring
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
        Calculate additional metrics based on information already in array

        :param metric_name: Key to save the metric under
        :param keys: One or multiple keys to calculate with.
            Each key will be used to index an array, ConditionArray[key].
            Each key should produce a 2D array when indexed.
        :param func: Name of numpy function to apply, e.g. 'sum'
        :param inverse: If True, repeats the calculation as if the keys
            were passed in the opposite order and saves in the other keys.
        :param propagate: If True, propagates the results of the calculation
            to the other keys in the array.
        :param frame_rng: Frames to use in calculation. If int, takes that
            many frames from start of trace. If tuple, uses passed
            frames.

        TODO:
            - Add possiblity for custom Callable function
            - Peaks could probably be passed here???
            - So could segmentation of peaks???
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
        Remove cells from array that do not match the filter.

        :param filter_name: Options are 'outside', 'inside',
            'outside_percentile', 'inside_percentile'.
        :param metric: Name of metric to use. Can be any key in the
            array.
        :param region: Name of region to calculate the filter in.
        :param channel: Name of channel to calculate filter in.
        :param frame_rng: Frames to use in calculation. If int, takes that
            many frames from start of trace. If tuple, uses passed
            frames.

        TODO:
            - Add ability to pass Callable, has to be done after Extract now
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
        skimage.regionprops or Extract._possible_metrics.

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
