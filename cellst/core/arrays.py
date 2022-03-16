import os
import warnings
import itertools
import functools
from typing import List, Tuple, Dict, Callable, Collection, Union

import h5py
import numpy as np
import plotly.graph_objects as go

import cellst.utils.filter_utils as filt
from cellst.utils.plot_utils import plot_groups
from cellst.utils.info_utils import nan_helper_2d
import cellst.utils.filter_utils as filtu
from cellst.utils.unet_model import UPeakModel
from cellst.utils.upeak.peak_utils import segment_peaks_agglomeration
from cellst.utils.metric_utils import active_cells, cumulative_active


class ConditionArray():
    """
    ax 0 - cell locations (nuc, cyto, population, etc.)
    ax 1 - channels (TRITC, FITC, etc.)
    ax 2 - metrics (median_int, etc.)
    ax 3 - cells
    ax 4 - frames
    """
    __slots__ = ('_arr', 'name', 'time', 'coords', '_arr_dim', '_dim_idxs',
                 '_key_dim_pairs', '_key_coord_pairs', 'masks', 'pos_id',
                 '__dict__')

    def __init__(self,
                 regions: List[str] = ['nuc'],
                 channels: List[str] = ['TRITC'],
                 metrics: List[str] = ['label'],
                 cells: List[int] = [0],
                 frames: List[int] = [0],
                 name: str = 'default',
                 time: Union[float, np.ndarray] = None,
                 pos_id: int = 0
                 ) -> None:
        """
        TODO:
            - Reorder if-statements for speed in key_coord functions.
            - Should save path to file in hdf5 as well for re-saving Conditions
        """
        # Convert inputs to tuple
        regions = tuple(regions)
        channels = tuple(channels)
        metrics = tuple(metrics)
        cells = tuple(cells)
        frames = tuple(frames)

        # Save some values
        self.name = name
        self.pos_id = pos_id
        self.masks = {}

        # Set _arr_dim based on input values - this can't change
        self._arr_dim = (len(regions), len(channels), len(metrics),
                         len(cells), len(frames))

        # Create empty data array
        self._arr = np.zeros(self._arr_dim)

        # Create coordinate dictionary
        self.coords = dict(regions=regions, channels=channels, metrics=metrics,
                           cells=cells, frames=frames)
        self._make_key_coord_pairs(self.coords)

        # Set time axis
        self.set_time(time)

    def __getitem__(self, key):
        # Needed if only one key is passed
        if not isinstance(key, tuple):
            key = tuple([key])
        # Sort given indices to the appropriate axes
        indices = self._convert_keys_to_index(key)

        return self._correct_output_dimensions(indices)

    def __setitem__(self, key, value):
        # Sort given indices to the appropriate axes
        if not isinstance(key, tuple):
            key = tuple([key])
        indices = self._convert_keys_to_index(key)

        self._arr[indices] = value

    def __str__(self):
        return self._arr.__str__()

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def coordinates(self):
        return tuple(self.coords.keys())

    @property
    def coordinate_dimensions(self):
        return {k: len(v) for k, v in self.coords.items()}

    @property
    def condition(self):
        return self.name

    @property
    def regions(self):
        return self.coords['regions']

    @property
    def channels(self):
        return self.coords['channels']

    @property
    def metrics(self):
        return self.coords['metrics']

    @property
    def keys(self):
        return list(itertools.product(self.coords['regions'],
                                      self.coords['channels'],
                                      self.coords['metrics']))

    @property
    def _is_empty(self) -> bool:
        """"""
        return any([not s for s in self.shape])

    def save(self, path: str) -> None:
        """
        Saves ConditionArray to an hdf5 file.

        TODO:
            - Add checking for path and overwrite options
        """
        f = h5py.File(path, 'w')
        f.create_dataset(self.name, data=self._arr)
        for coord in self.coords:
            # Axis names and coords stored as attributes
            if coord in ('frames', 'cells'):
                if isinstance(self.coords[coord], (int, float)):
                    pass
                elif not len(self.coords[coord]):
                    self.coords[coord] = np.array(self.coords[coord])
                elif np.array((self.coords[coord])).max() >= 2 ** 16:
                    raise ValueError('Cannot save values larger than 16-bit.')

                f[self.name].attrs[coord] = np.array(self.coords[coord],
                                                     dtype=np.uint16)
            else:
                f[self.name].attrs[coord] = self.coords[coord]

        f[self.name].attrs['pos_id'] = self.pos_id
        f[self.name].attrs['time'] = self.time

        f.close()

    @classmethod
    def load(cls, path: str) -> None:
        """
        Load a structured arrary and convert to sites dict.

        TODO:
            - Add a check that path exists
        """
        f = h5py.File(path, "r")
        return cls._build_from_file(f)

    @classmethod
    def _build_from_file(cls, f: h5py.File) -> 'ConditionArray':
        """
        Given an hdf5 file, returns a ConditionArray instance
        """
        if len(f) != 1:
            raise TypeError('Too many keys in hdf5 file.')
        for key in f:
            _arr = ConditionArray(**f[key].attrs, name=key)
            _arr[:] = f[key]

        return _arr

    def _getitem_w_idx(self, idx):
        """
        Index CustomArray w/o recalculating the indices each time
        """
        return self._correct_output_dimensions(idx)

    def _correct_output_dimensions(self,
                                   idx: Tuple[slice, str]
                                   ) -> np.ndarray:
        """Output must be at least 2D, but no other axes of len 1"""
        out = np.squeeze(self._arr[idx])
        if out.ndim == 1:
            # Figure out how many cells/frames total
            _cidx, _fidx = self._dim_idxs['cells'], self._dim_idxs['frames']
            tot_cells = self._arr.shape[_cidx]
            tot_frames = self._arr.shape[_fidx]

            # Index a pretend array to figure out how many were requested
            if isinstance(idx[_cidx], int):
                req_cells = 1
            else:
                req_cells = len(np.empty(tot_cells)[idx[_cidx]])
            if isinstance(idx[_fidx], int):
                req_frames = 1
            else:
                req_frames = len(np.empty(tot_frames)[idx[_fidx]])
            cf = [req_cells, req_frames]
            missing = [s not in out.shape for s in cf]

            # Need to add either cells or frames back
            if all(missing):
                # Add to last axis
                # TODO: Not sure what makes the most sense here
                out = np.expand_dims(out, -1)
            elif any(missing):
                out = np.expand_dims(out, missing.index(True))

        return out

    def _convert_keys_to_index(self,
                               key: Tuple[(str, slice)]
                               ) -> Tuple[(int, slice)]:
        """
        Converts strings and slices given in key to the saved
        dimensions and coordinates of self._xarr.

        Args:
            - key:

        Returns:
            Converted keys containing no strings

        NOTE: integer indices are best provided last. If they are
              provided first, they will possibly get overwritten if
              too few keys were provided.

        TODO:
            - Add handling of Ellipsis in key
        """
        # Check that key is not too long
        if len(key) > len(self.coords):
            raise ValueError(f'Max number of dimensions is {len(self.coords)}.'
                             f' Got {len(key)}.')

        # Get dimensions for the keys
        indices = [slice(None)] * len(self.coords)
        cell_idx = self._dim_idxs['cells']
        frame_idx = self._dim_idxs['frames']
        seen_int = 0  # This will be used to separate cells and frames
        # Used to check slice types
        i_type = (int, type(None))
        s_type = (str, type(None))

        # Sort each key
        for k_idx, k in enumerate(key):
            if isinstance(k, slice):
                if isinstance(k.start, type(None)) and isinstance(k.stop, type(None)):
                    # All none means user had to put it in the correct spot
                    idx = k_idx
                    start_coord = k.start
                    stop_coord = k.stop
                    # Count it if it is in the cell axis
                    if k_idx == cell_idx:
                        seen_int += 1
                elif isinstance(k.start, i_type) and isinstance(k.stop, i_type):
                    # Integer slice - check to see if frames or cells
                    if seen_int == 1:
                        idx = frame_idx
                    elif seen_int == 0:
                        idx = cell_idx
                        seen_int += 1
                    else:
                        raise ValueError('Max number of integer indices is 2.')

                    start_coord = k.start
                    stop_coord = k.stop

                elif isinstance(k.start, s_type) and isinstance(k.stop, s_type):
                    # String slice
                    # First check that start and step are in the same dim
                    try:
                        start_dim = None if not k.start else self._key_dim_pairs[k.start]
                        stop_dim = None if not k.stop else self._key_dim_pairs[k.stop]
                    except KeyError:
                        raise KeyError(f'Some of {k.start, k.stop} were not found '
                                       'in any dimension.')
                    if start_dim and stop_dim and (start_dim != stop_dim):
                        raise IndexError(f"Dimensions don't match: {k.start} is in "
                                         f"{start_dim}, {k.stop} is in {stop_dim}.")

                    # Get axis and coord indices and remake the slice
                    idx = self._dim_idxs[start_dim]
                    start_coord = self._key_coord_pairs[k.start] if k.start else None
                    stop_coord = self._key_coord_pairs[k.stop] if k.stop else None

                else:
                    # Mixed slice
                    raise ValueError('Mixing integers and strings in slices is not allowed.')

                indices[idx] = slice(start_coord, stop_coord, k.step)

            elif isinstance(k, str):
                # Easiest case - save str for indexing
                idx = self._dim_idxs[self._key_dim_pairs[k]]
                indices[idx] = self._key_coord_pairs[k]

            elif isinstance(k, int):
                # Integer - check to see if frames or cells
                if seen_int == 1:
                    indices[frame_idx] = k
                elif seen_int == 0:
                    indices[cell_idx] = k
                    seen_int += 1
                else:
                    raise ValueError('Max number of integer indices is 2.')

            elif isinstance(k, (tuple, list, np.ndarray)):
                # Select arbitrary indices in a single axis (i.e. multiple metrics)
                # Save all values and idxs
                new = []
                idx = []
                for item in k:
                    # Treat individual items as we would above
                    if isinstance(item, str):
                        new.append(self._key_coord_pairs[item])
                        idx.append(self._dim_idxs[self._key_dim_pairs[item]])
                    elif isinstance(item, int):
                        new.append(item)
                    else:
                        raise ValueError(f'Did not understand key {item} in {k}.')

                # Check that we aren't trying to index multiple axes
                idx = np.unique(idx)
                if len(idx) == 0:
                    idx = k_idx
                elif len(idx) > 1:
                    raise IndexError(f'Indices in {k} map to multiple axes.')
                else:
                    idx = idx[0]

                indices[idx] = np.array(new, dtype=int)

            else:
                raise ValueError(f'Indices must be int, str, or slice. Got {type(k)}.')

        return tuple(indices)

    def _make_key_coord_pairs(self, coords: dict) -> None:
        """
        """
        # Convert string axis index to integer index
        self._dim_idxs = {k: n for n, k in enumerate(coords.keys())}

        # Check for duplicate keys (allowed in cell and frame)
        to_check = ['regions', 'channels', 'metrics']
        all_poss = [sl for l in [coords[t] for t in to_check] for sl in l]
        if len(all_poss) != len(set(all_poss)):
            raise KeyError(f'All coordinates in dimensions {to_check} must be '
                           f'unique. Got {all_poss}.')

        # Match keys and coordinates
        self._key_dim_pairs = {
            a: [k for k, v in coords.items() if a in tuple(v)][0]
            for a in all_poss
        }
        # NOTE: Not sure I will need this one
        self._key_coord_pairs = {
            a: [v.index(a) for k, v in coords.items() if a in tuple(v)][0]
            for a in all_poss
        }

    def _get_key_components(self,
                            key: Tuple[int, str],
                            exclude: str = 'metrics'
                            ) -> Tuple[int, str]:
        """Returns the components in key that are NOT in the target axis"""
        assert exclude in self.coords, f'Coordinate {exclude} not found.'

        # Get axis for each key and compare to exclude
        out = []
        for k in key:
            coord = self._key_dim_pairs[k]
            if coord != exclude:
                out.append(k)

        return tuple(out)

    def set_position_id(self, pos: int = None) -> None:
        """
        Adds unique identifiers for cells in ConditionArray

        TODO:
            - Catch TypeError, ValueError for non-digit pos
        """
        pos = pos if pos else int(self.pos_id)

        # Create the position id
        arr = np.ones((self._arr_dim[self._dim_idxs['cells']],
                       self._arr_dim[self._dim_idxs['frames']]))
        arr *= pos

        # Expand metric by 1
        if 'position_id' not in self.coords['metrics']:
            self.add_metric_slots('position_id')

        self.__setitem__('position_id', arr)

    def add_metric_slots(self, name: List[str]) -> None:
        """
        This needs to expand self._arr along the metric axis
        to make room for an additional metric
        """
        # Format inputs
        if isinstance(name, str):
            name = [name]

        # Remove names that have already been added
        name = [n for n in name if n not in self.coords['metrics']]
        # If all names are gone - give up
        if not name: return

        # Get dimensions and set the metric dimension to 1
        new_dim = list(self._arr_dim)
        new_dim[self._dim_idxs['metrics']] = len(name)

        # Concatenate a new array on the metric axis
        new_arr = np.empty(new_dim).astype(float)
        new_arr[:] = np.nan
        self._arr = np.concatenate((self._arr, new_arr),
                                   axis=self._dim_idxs['metrics'])

        # Update all the necessary attributes
        self._arr_dim = self._arr.shape
        self.coords.update(dict(metrics=tuple([*self.coords['metrics'], *name])))

        # Recalculate the coords
        self._make_key_coord_pairs(self.coords)

    def filter_cells(self,
                     mask: np.ndarray = None,
                     key: str = None,
                     delete: bool = True,
                     *args, **kwargs
                     ) -> np.ndarray:
        """
        Either uses an arbitrary mask or a saved mask (key) to
        filter the data. If delete, the underlying structure
        is changed, otherwise, the data are only returned.

        TODO:
            - Add option to return a new Condition instead of
              an np.ndarray
        """
        # If key is provided, look for saved mask
        if key and mask is None:
            mask = self.masks[key]
        elif not key and mask is None:
            warnings.warn('Did not get mask or key. Nothing done.',
                          UserWarning)
            return

        # Check that mask is bool/int
        if mask.dtype not in (int, bool):
            raise TypeError(f'Mask must be bool or int. Got {mask.dtype}')

        # Make sure mask is the correct dimension
        indices = self.reshape_mask(mask)

        if delete:
            # Delete items from self._arr and any existing masks
            self._arr = self._arr[indices]
            self._arr_dim = self._arr.shape
            for k, v in self.masks.items():
                self.masks[k] = v[indices]

            # Recalculate the cell coords
            # TODO: Should there be a check that dimensions are equal?
            if mask.ndim == 2:
                mask = mask.any(1)
            self.coords['cells'] = tuple([
                cell for cell, keep in zip(self.coords['cells'], mask)
                if keep])

            return self._arr

        return self._arr[mask]

    def reshape_mask(self,
                     mask: np.ndarray
                     ) -> Tuple[slice, type(Ellipsis), np.ndarray]:
        """
        Takes in a 1D, 2D, or 5D mask and casts to tuple of
        5 dimensional indices. Use this to apply a 1D mask
        to self._arr.

        Assumes that the only thing being filtered is cells.
        """
        # new_mask = np.ones(self.shape).astype(bool)
        if mask.ndim == 1:
            # Assume it applies to cell axis
            return tuple([Ellipsis, mask, slice(None)])
        elif mask.ndim == 2:
            # Assume it applies to cell and frame axes
            # Note sure that this makes sense
            return tuple([Ellipsis, mask.any(1), slice(None)])
        elif mask.ndim == 5:
            # TODO: Not sure how to handle this or if it
            #       is even possible to index w/o losing shape
            return mask
        else:
            raise ValueError('Dimensions of mask must be 1, 2, or 5. '
                             f'Got {mask.ndim}.')

    def remove_parents(self,
                       parent_daughter: Dict,
                       cell_index: Dict
                       ) -> np.ndarray:
        """
        Returns 1D boolean mask to remove parent cells

        TODO:
            - Add option to create cell_index from track
        """
        # Find indices of all parents along cell axis
        parents = np.unique(tuple(parent_daughter.values()))
        parent_idx = [cell_index[p] for p in parents]

        # Make the mask
        mask = np.ones(len(cell_index)).astype(bool)
        mask[parent_idx] = False

        return mask

    def remove_short_traces(self, min_trace_length: int) -> np.ndarray:
        """
        Removes cells with less than min_trace_length non-nan values
        """
        mask = np.ones(self.shape[self._dim_idxs['cells']]).astype(bool)

        # Count non-nans in each trace
        # Label is always first metric
        nans = ~np.isnan(self._arr[0, 0, 0])
        counts = np.sum(nans, axis=1)

        # Mask cells with too few values
        mask[counts < min_trace_length] = False

        return mask

    def generate_mask(self,
                      function: [Callable, str],
                      metric: str,
                      region: [str, int] = 0,
                      channel: [str, int] = 0,
                      frame_rng: (Tuple[int], int) = None,
                      key: str = None,
                      *args, **kwargs
                      ) -> np.ndarray:
        """
        I want this function to be able to generate arbitrary
        masks from some default functions
        i.e. percentiles, abs_val, etc..

        frame_rng only applies to filter_utils right now
        """
        # Format inputs to the correct type
        if isinstance(region, int):
            region = self.coords['regions'][region]
        if isinstance(channel, int):
            channel = self.coords['channels'][channel]

        # Extract data for a single metric
        vals = self[region, channel, metric, :, :]

        if isinstance(function, Callable):
            # Call user function to get mask
            mask = function(vals, *args, **kwargs)
        elif isinstance(function, str):
            # Else mask should come from the filter utils
            try:
                '''
                There is a fundamental flaw to using nans to mask the frames
                as elegant as it seems. The issue is if the frames in question
                have nans in the data. If the user is expecting ignore_nans to
                be False, those cells should be removed, but in this case, they
                are not.
                '''
                user_mask = np.zeros_like(vals).astype(bool)
                idx = None
                # Select the opposite of the given idx to mark as nan
                if isinstance(frame_rng, (int, float)):
                    if frame_rng < 0:
                        # If negative, all frames before it
                        idx = slice(None, frame_rng)
                    else:
                        # If positive, all frames after it
                        idx = slice(frame_rng, None)
                elif isinstance(frame_rng, (tuple, list)):
                    # If range, all other values
                    assert len(frame_rng) == 2
                    val_frames = np.arange(vals.shape[1])
                    idx_frames = np.arange(*frame_rng)
                    idx = ~np.in1d(val_frames, idx_frames)
                elif isinstance(frame_rng, type(None)):
                    # Using all the frames
                    pass
                else:
                    warnings.warn(f'Could not use frame_rng {frame_rng}, '
                                  'Using all frames by default.',
                                  UserWarning)
                # Mark frames to be removed
                if idx:
                    user_mask[:, idx] = True

                mask = getattr(filtu, function)(vals, mask=user_mask,
                                                *args, **kwargs)
            except AttributeError:
                raise AttributeError('Did not understand filtering '
                                     f'function {function}.')

        if key is not None:
            # Save the mask if key is given
            self.masks[key] = mask

        return mask

    def get_mask(self, key: str) -> np.ndarray:
        return self.masks[key]

    def set_time(self, time: float) -> None:
        """
        Define the time axis. Time is the interval between frames
        """
        if time is None:
            self.time = self.coords['frames']
        elif isinstance(time, np.ndarray):
            self.time = time
        else:
            self.time = np.arange(len(self.coords['frames'])) * time

    def set_condition(self, condition: str) -> None:
        """
        Updates name of the ConditionArray.
        """
        self.name = condition

    def propagate_values(self,
                         key: Tuple[str],
                         prop_to: str = 'both'
                         ) -> None:
        """Propagates metric value to other keys"""
        assert isinstance(key, tuple)
        assert len(key) == 3

        # Get original key values
        data = self[key]
        dims = {self._key_dim_pairs[k]: k for k in key}
        chan = dims['channels']
        regn = dims['regions']
        metr = [dims['metrics']]

        # Figure out where to propagate
        if prop_to in ('c', 'chnl', 'channel'):
            # Get all other channels
            _chan = [c for c in self.channels if c != chan]
        elif prop_to in ('r', 'rgn', 'region'):
            # Get all other regions
            _regn = [r for r in self.regions if r != regn]
        else:
            _chan = [c for c in self.channels if c != chan]
            _regn = [r for r in self.regions if r != regn]
        chan = _chan if _chan else [chan]
        regn = _regn if _regn else [regn]

        # Build keys and assign values
        new_keys = itertools.product(chan, regn, metr)
        for nk in new_keys:
            self[nk] = data

    def interpolate_nans(self, keys: Collection[tuple] = None) -> None:
        """Linear interpolation of nans in each row

        Args:

        Returns:
        """
        if not keys:
            keys = self.keys

        for k in keys:
            k = tuple(k)
            self[k] = nan_helper_2d(self[k])

    def predict_peaks(self,
                      key: Tuple[int, str],
                      model: UPeakModel = None,
                      weight_path: str = 'cellst/config/upeak_example_weights.tf',
                      propagate: bool = True,
                      segment: bool = True,
                      **kwargs
                      ) -> None:
        """"""
        # Get the data that will be used for prediction
        assert isinstance(key, tuple)
        data = self[key]
        assert data.ndim == 2

        # Make the destination metric slots and keys
        slots = ['slope_prob', 'plateau_prob']
        if segment:
            # Add the extra slot here
            self.add_metric_slots(slots + ['peaks'])
        else:
            self.add_metric_slots(slots)

        base = self._get_key_components(key, 'metrics')
        dest_keys = [base + tuple([s]) for s in slots]

        # Initialize the UPeak model if needed
        if not model:
            model = UPeakModel(weight_path)

        # Get predictions of where peaks exist
        predictions = model.predict(data, roi=(1, 2))  # slope, plateau
        for i, d in enumerate(dest_keys):
            self[d] = predictions[..., i]
            if propagate: self.propagate_values(d, prop_to=propagate)

        # Segment peaks if needed
        if segment:
            k = base + ('peaks',)
            peaks = segment_peaks_agglomeration(data, predictions, **kwargs)
            self[k] = peaks
            if propagate: self.propagate_values(k, prop_to=propagate)

    def mark_active_cells(self,
                          key: Tuple[int, str],
                          thres: float = 1,
                          propagate: bool = True
                          ) -> None:
        """"""
        # Get the data that will be used for prediction
        assert isinstance(key, tuple)
        data = self[key]
        assert data.ndim == 2

        # Make the destination slots and keys
        slots = ['active', 'cumulative_active']
        self.add_metric_slots(slots)
        base = self._get_key_components(key, 'metrics')
        dest_keys = [base + tuple([s]) for s in slots]

        # Calculate the active cells
        active = active_cells(data, thres)
        cumul_active = cumulative_active(active)
        self[dest_keys[0]] = active
        self[dest_keys[1]] = cumul_active

        if propagate:
            [self.propagate_values(d, prop_to=propagate)
             for d in dest_keys]


class ExperimentArray():
    """
    TODO:
        - Add Typing hints when the imports are fixed
    """
    __slots__ = ('name', 'attrs', 'sites', 'masks', '__dict__')

    class _CondIndexer():
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            # Return ConditionArray if int or slice
            if isinstance(key, int):
                return self.data[key]
            elif isinstance(key, slice):
                return self.data[key]
            elif not isinstance(key, tuple):
                key = tuple([key])

            # Otherwise, return FROM each ConditionArray
            indices = self.data[0]._convert_keys_to_index(key)
            return [v._getitem_w_idx(indices) for v in self.data]

    def __init__(self,
                 arrays: List[ConditionArray] = None,
                 name: str = None,
                 time: float = None,
                 ) -> None:
        # Save some values
        self.name = name
        self.sites = {}

        # Save arrays if given
        if arrays is not None:
            [self.__setitem__(None, a) for a in arrays]

        # Build dictionary for saving masks
        self.masks = {k: {} for k in self.sites}

    def __setitem__(self, key, value):
        # All values must be ConditionArray
        if not isinstance(value, ConditionArray):
            raise TypeError('All values in ExperimentArray must be'
                            f'type ConditionArray. Got {type(value)}.')

        self.sites[key] = value

    def __getitem__(self, key):
        """
        Assumption is that all Arrays have the same coordinates.
        The first Array is used to generate the dictionaries for indexing
        """
        try:
            # First try to return a site the user requested
            if isinstance(key, (tuple, list)):
                return self._CondIndexer([self.sites[k] for k in key])
            else:
                return self.sites[key]
        except KeyError:
            # If site doesn't exist, try using key to index all sites
            if not isinstance(key, tuple):
                key = tuple([key])
            indices = tuple(self.sites.values())[0]._convert_keys_to_index(key)

            return [v._getitem_w_idx(indices) for v in self.sites.values()]

    def __len__(self):
        return len(self.sites)

    def __str__(self):
        return str(self.sites)

    @property
    def shape(self):
        return [v.shape for v in self.sites.values()]

    @property
    def ndim(self):
        return [v.ndim for v in self.sites.values()]

    @property
    def dtype(self):
        return [v.dtype for v in self.sites.values()]

    @property
    def conditions(self) -> List:
        return [v.name for v in self.sites.values()]

    @property
    def regions(self):
        return [v.regions for v in self.sites.values()]

    @property
    def channels(self):
        return [v.channels for v in self.sites.values()]

    @property
    def metrics(self):
        return [v.metrics for v in self.sites.values()]

    @property
    def time(self):
        return [v.time for v in self.sites.values()]

    @property
    def coordinates(self):
        return tuple(next(iter(self.values())).coords.keys())

    def items(self):
        return self.sites.items()

    def values(self):
        return self.sites.values()

    def keys(self):
        return self.sites.keys()

    def update(self, *args, **kwargs):
        return self.sites.update(*args, **kwargs)

    def set_time(self, time: float = None) -> None:
        """
        Define the time axis
        """
        for v in self.sites.values():
            v.set_time(time)

    def set_conditions(self, condition_map: Dict[str, str] = {}) -> None:
        """
        Updates name of Condition arrays in Experiment.
        condition_map should map Condition.name to desired condition.
        """
        for k, v in condition_map.items():
            try:
                self.sites[k].set_condition(v)
            except KeyError:
                # TODO: Should this warn users?
                pass

    def load_condition(self,
                       path: str,
                       name: str = None,
                       pos_id: int = None,
                       ) -> None:
        """
        Used to add a Condition to experiment directly from an hdf5 file
        Saves as name + pos_id

        TODO:
            - Add function to walk dirs, and load hdf5 files, with uniq names
                See Orchestrator.build_experiment_file()
        """
        # Get array and key
        arr = ConditionArray.load(path)
        name = name if name else arr.name
        pos_id = pos_id if pos_id else arr.pos_id
        if pos_id:
            key = f'{name}{pos_id}'
        else:
            key = name

        self.__setitem__(key, arr)

    def save(self, path: str) -> None:
        """Saves all the Conditions in Experiment to an hdf5 file.

        Loads the hdf5 file for each condition and then saves them
        in a single hdf5 file at path. Runs merge_conditions() first
        to ensure data doesn't get overwritten.

        Args:
            path: Path to the location where file should be saved

        Returns:
            None

        Raises:
            ValueError: If any cell or frame is greater than 2 ** 16
        """
        '''
        TODO:
            - Add checking for path and overwrite options
            - Add low memory option
        '''
        f = h5py.File(path, 'w')
        self.merge_conditions()
        for key, val in self.sites.items():
            # Array data stored as a dataset
            f.create_dataset(key, data=val._arr)

            # Axis names and coords stored as attributes
            for coord in val.coords:
                # Cells and frames have the potential to be large, so handle separately
                if coord in ('frames', 'cells'):
                    if isinstance(val.coords[coord], (int, float)):
                        pass
                    elif not len(val.coords[coord]):
                        val.coords[coord] = np.array(val.coords[coord])
                    elif np.array((val.coords[coord])).max() >= 2 ** 16:
                        raise ValueError('Cannot save values larger than 16-bit.')

                    f[key].attrs[coord] = np.array(val.coords[coord], dtype=np.uint16)
                else:
                    f[key].attrs[coord] = val.coords[coord]

        f.close()

    @classmethod
    def load(cls, path: str) -> None:
        """
        Load a structured arrary and convert to sites dict.

        TODO:
            - Add a check that path exists
        """
        f = h5py.File(path, "r")
        return cls._build_from_file(f)

    def remove_empty_sites(self) -> None:
        """Removes all sites that have any empty dimension"""
        self.sites = {k: v for k, v in self.sites.items()
                      if not v._is_empty}

    def remove_short_traces(self, min_trace_length: int = 0) -> None:
        """Applies a filter to each condition to remove cells with
        fewer good frames than min_trace_length

        Args:
            min_trace_length: Minimum number of frames allowed

        Returns:
            None
        """
        masks = [v.remove_short_traces(min_trace_length)
                 for v in self.sites.values()]

        for m, v in zip(masks, self.sites.values()):
            v.filter_cells(m, delete=True)

    def generate_mask(self,
                      function: [Callable, str],
                      metric: str,
                      region: [str, int] = 0,
                      channel: [str, int] = 0,
                      frame_rng: (Tuple[int], int) = None,
                      key: str = None,
                      individual: bool = True,
                      *args, **kwargs
                      ) -> np.ndarray:
        """Generates a boolean mask for each Condition

        Args:
            function:
            metric:
            region:
            channel:
            frame_rng:
            key:
            *args:
            **kwargs:

        Returns:
            np.ndarray
        """
        # Format inputs to the correct type
        if isinstance(region, int):
            region = self.regions[0][region]
        if isinstance(channel, int):
            channel = self.channels[0][channel]

        if individual:
            # Build masks in each Condition with the function
            _masks = [v.generate_mask(function, metric, region,
                                      channel, frame_rng, key,
                                      *args, **kwargs)
                      for v in self.sites.values()]
        else:
            # Build data for all of the sites together
            # Vals should be 2D for all inputs
            vals = self[region, channel, metric, :, :]
            splits = get_split_idxs(vals, axis=0)

            # Get mask and split them up
            # TODO: Only works for strings
            function = getattr(filtu, function)
            masks = function(np.vstack(vals), *args, **kwargs)
            _masks = split_array(masks, splits, axis=0)

        # Save if needed
        if key is not None:
            self.masks[key] = _masks

        return _masks

    def filter_cells(self,
                     mask: List[np.ndarray] = None,
                     key: str = None,
                     delete: bool = True,
                     *args, **kwargs
                     ) -> np.ndarray:
        """
        Either uses an arbitrary mask or a saved mask (key) to
        filter the data. If delete, the underlying structure
        is changed, otherwise, the data are only returned.

        TODO:
            - Add option to return a new Condition instead of
              an np.ndarray
        """
        # If key is provided, look for saved mask
        if mask is None and key is not None:
            mask = self.masks[key]
        elif mask is None and key is None:
            warnings.warn('Did not get mask or key. Nothing done.',
                          UserWarning)
            return

        # Make sure enough lists are available
        if isinstance(mask, np.ndarray):
            mask = [mask]
        # Lengthen mask list if needed
        if len(mask) == 1:
            mask = mask * len(self.sites)
        elif len(mask) != len(self.sites):
            raise ValueError(f'Have {len(self.sites)} sites and '
                             f'{len(mask)} masks.')

        # Mask type and dimension will be checked in Condition
        out = [arr.filter_cells(msk, key, delete, *args, **kwargs)
               for msk, arr in zip(mask, self.sites.values())]

        return out

    def add_metric_slots(self, name: List[str]) -> None:
        """"""
        for v in self.sites.values():
            v.add_metric_slots(name)

    def merge_conditions(self) -> None:
        """
        Concatenate Conditions with matching conditions

        TODO:
            - Add a way to pass lists of keys to merge
            - Difflib might be a way to handle this...
            - Saved masks should also be concatenated and saved
        """
        # Get the unique conditions and respective arrs
        _nm = lambda x: x.name
        uniq_conds = []
        cond_arr_grps = []
        for k, g in itertools.groupby(sorted(self.values(), key=_nm), _nm):
            uniq_conds.append(k)
            cond_arr_grps.append(list(g))

        need_merge = len(uniq_conds) != len(self)
        if need_merge:
            for cond, cond_arrs in zip(uniq_conds, cond_arr_grps):
                if len(cond_arrs) <= 1:
                    # Skip merging for these conditions
                    continue

                # Need to add position ID to keep unique identification
                if len(set([c.pos_id for c in cond_arrs])) < len(cond_arrs):
                    for n, c in enumerate(cond_arrs):
                        # Try to guess pos_id, or just count
                        try:
                            pos = int(c.name[len(cond):])
                        except ValueError:
                            # TODO: Probably raise a warning here
                            pos = n

                        c.set_position_id(n)

                coords = cond_arrs[0].coords
                # TODO: Kinda messy and takes a while... - is there a better place
                if 'position_id' not in coords['metrics']:
                    [c.set_position_id() for c in cond_arrs]

                # cells are always indexed by integer, so make new list
                coords['cells'] = np.arange(sum((len(c.coords['cells'])
                                            for c in cond_arrs)))

                # Concatenate the arrays along cell axis
                ax = cond_arrs[0]._dim_idxs['cells']
                new_arr = np.concatenate([c._arr for c in cond_arrs], axis=ax)

                # Delete old arrays
                keys_to_delete = [k for k, v in self.items()
                                  if cond == v.name]
                for k in keys_to_delete:
                    self.sites.pop(k, None)

                # Save new one
                self.sites[cond] = ConditionArray(**coords, name=cond,
                                                  time=cond_arrs[0].time)
                self.sites[cond][:] = new_arr

    def interpolate_nans(self, keys: Collection[tuple] = None) -> None:
        """Linear interpolation of nans in each row

        Args:

        Returns:
        """
        for v in self.sites.values():
            v.interpolate_nans(keys)

    def predict_peaks(self,
                      key: Tuple[int, str],
                      weight_path: str = 'cellst/config/upeak_example_weights.tf',
                      propagate: bool = True,
                      segment: bool = True,
                      **kwargs
                      ) -> None:
        """
        kwargs are passed to the segmentation algorithm
        TODO: Initialize the UPeak model here and pass to the sites
        """
        '''
        NOTE: This will fail if any of the sites have different dimensions
        This is important to remember if adding groups to Arrays.
        Follows same assumption as made for keys - i.e. all are the same
        '''
        model = UPeakModel(weight_path)
        for v in self.sites.values():
            v.predict_peaks(key, model, propagate=propagate)

    def mark_active_cells(self,
                          key: Tuple[int, str],
                          thres: float = 1,
                          propagate: bool = True
                          ) -> None:
        """"""
        for v in self.sites.values():
            v.mark_active_cells(key, thres, propagate)

    def plot_by_condition(self,
                          keys: List[Tuple[str]],
                          conditions: Collection[str] = None,
                          estimator: Union[Callable, str, functools.partial] = None,
                          err_estimator: Union[Callable, str, functools.partial] = None,
                          kind: str = 'line',
                          title: str = None,
                          x_label: str = None,
                          y_label: str = None,
                          x_limit: Tuple[float] = None,
                          y_limit: Tuple[float] = None,
                          layout_spec: dict = {},
                          show: bool = False,
                          save: str = None,
                          _format: str = 'svg'
                          ) -> go.Figure:
        """
        rename to be like time_plot or something
        keys must return a 2D arr for this to work.
        TODO: More generic way to group conditions
        TODO: Add option to save figures
        """
        # Get the data to plot
        keys = tuple(keys) if not isinstance(keys, tuple) else keys
        conditions = conditions if conditions else self.conditions

        # Get the data to plot
        arrs = self[conditions][keys]
        time = self.time[0]

        # Make the base plot
        fig = plot_groups(arrs, conditions, estimator, err_estimator, kind=kind, time=time)

        # Update the figure layout
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(title=y_label, range=y_limit)
        fig.update_layout(title=title, **layout_spec)

        if show:
            fig.show()
        if save:
            # Determines fig type based on extension
            html = save.split('.')[-1] == 'html'
            if html:
                config = {'toImageButtonOptions': {
                            'format': _format,
                            'filename': 'figure',
                            'scale': 1
                            }
                         }
                fig.write_html(save, config=config)
            else:
                fig.write_image(save)

        return fig

    @classmethod
    def _build_from_file(cls, f: h5py.File) -> 'ExperimentArray':
        """
        Given an hdf5 file, returns a ExperimentArray instance
        """
        pos = ExperimentArray()
        for key in f:
            # Attrs define the coords and axes
            _arr = ConditionArray(**f[key].attrs, name=key)
            # f[key] holds the actual array data
            _arr[:] = f[key]
            pos[key] = _arr

        return pos
