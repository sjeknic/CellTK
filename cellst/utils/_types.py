import warnings
from typing import NewType, Collection, Tuple, Callable, Dict

import h5py
import numpy as np

import cellst.utils.filter_utils as filt


class CellArray():
    """
    ax 0 - cell locations (nuc, cyto, population, etc.)
    ax 1 - channels (TRITC, FITC, etc.)
    ax 2 - metrics (median_int, etc.)
    ax 3 - cells
    ax 4 - frames
    """
    __slots__ = ('_arr', 'name', 'attrs', 'coords', '_arr_dim', '_dim_idxs',
                 '_key_dim_pairs', '_key_coord_pairs', '_nan_mask')

    def __init__(self,
                 regions: Collection[str] = ['nuc'],
                 channels: Collection[str] = ['TRITC'],
                 metrics: Collection[str] = ['label'],
                 cells: Collection[int] = [0],
                 frames: Collection[int] = [0],
                 name: str = 'default',
                 attrs: dict = None,
                 **kwargs
                 ) -> None:
        """
        TODO:
            - Handling of metrics which will have multiple entries (i.e. bbox-0, bbox-1, etc.)
              The simplest might be to just automatically convert the keys when the array is loaded.
              The issue with this is that this would also have to happen in __getitem__, which
              seems like it would slow things down. One option could be to check if any of those metrics
              exist when the CustomArray is made. If they are __getitem__ first has to call a function to
              sort that out, and if not, that function could just be a pass through function.
            - Reorder if-statements for speed in key_coord functions.
            - Add ability to save time steps
            - Add ability to expand dimensions and add new derived metrics
        """
        # Convert inputs to tuple
        regions = tuple(regions)
        channels = tuple(channels)
        metrics = tuple(metrics)
        cells = tuple(cells)
        frames = tuple(frames)

        # Save some values
        self.name = name
        self.attrs = attrs
        self._nan_mask = {}

        # Set _arr_dim based on input values - this can't change
        self._arr_dim = (len(regions), len(channels), len(metrics),
                         len(cells), len(frames))

        # Create empty data array
        self._arr = np.zeros(self._arr_dim)

        # Create coordinate dictionary
        self.coords = dict(regions=regions, channels=channels, metrics=metrics,
                           cells=cells, frames=frames)
        self._make_key_coord_pairs(self.coords)

    def __getitem__(self, key):
        # Needed if only one key is passed
        if not isinstance(key, tuple):
            key = tuple([key])
        # Sort given indices to the appropriate axes
        indices = self._convert_keys_to_index(key)

        return self._arr[indices]

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

    def save(self, path: str) -> None:
        """
        Saves CellArray to an hdf5 file.

        TODO:
            - Add checking for path and overwrite options
        """
        f = h5py.File(path, 'w')
        f.create_dataset(self.name, data=self._arr)
        for coord in self.coords:
            # Axis names and coords stored as attributes
            f[self.name].attrs[coord] = self.coords[coord]

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

    @staticmethod
    def _build_from_file(f: h5py.File) -> 'CellArray':
        """
        Given an hdf5 file, returns a CellArray instance
        """
        if len(f) != 1:
            raise TypeError('Did not understand hdf5 file format.')

        for key in f:
            _arr = CellArray(**f[key].attrs, name=key)
            _arr[:] = f[key]

        return _arr

    def _getitem_w_idx(self, idx):
        """
        Used by CustomSet to index CustomArray w/o recalculating
        the indices each time
        """
        return self._arr[idx]

    def _convert_keys_to_index(self, key) -> Tuple[(int, slice)]:
        """
        Converts strings and slices given in key to the saved
        dimensions and coordinates of self._xarr.

        Args:
            - key: tuple(slices)

        Returns:
            tuple(slices)

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
        # Five dimensions possible
        indices = [slice(None)] * len(self.coords)
        # TODO: I feel like names shouldn't be hard-coded
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
                        start_dim = None if k.start is None else self._key_dim_pairs[k.start]
                        stop_dim = None if k.stop is None else self._key_dim_pairs[k.stop]
                    except KeyError:
                        raise KeyError(f'Some of {k.start, k.stop} were not found '
                                       'in any dimension.')
                    if (start_dim is not None and
                        stop_dim is not None and
                        start_dim != stop_dim):
                        raise IndexError(f"Dimensions don't match: {k.start} is in "
                                         f"{start_dim}, {k.stop} is in {stop_dim}.")

                    # Get axis and coord indices and remake the slice
                    idx = self._dim_idxs[start_dim]
                    start_coord = self._key_coord_pairs[k.start] if k.start is not None else None
                    stop_coord = self._key_coord_pairs[k.stop] if k.stop is not None else None

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

    def add_metric_slots(self, name: Collection[str]) -> None:
        """
        This needs to expand self._arr along the metric axis
        to make room for an additional metric
        """
        # Format inputs
        if isinstance(name, str):
            name = [name]

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
            - Add option to return a new CellArray instead of
              an np.ndarray
        """
        # If key is provided, look for saved mask
        if mask is None and key is not None:
            mask = self._nan_mask[key]
        elif mask is None and key is None:
            warnings.warn('Did not get mask or key. Nothing done.',
                          UserWarning)
            return

        # Check that mask is bool/int
        if mask.dtype != bool and mask.dtype != int:
            raise TypeError(f'Mask must be bool or int. Got {mask.dtype}')

        # Make sure mask is the correct dimension
        indices = self.reshape_mask(mask)

        if delete:
            # Delete items from self._arr and any existing masks
            self._arr = self._arr[indices]
            self._arr_dim = self._arr.shape
            for k, v in self._nan_mask.items():
                self._nan_mask[k] = v[indices]

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

    def generate_mask(self,
                      function: [Callable, str],
                      metric: str,
                      region: [str, int] = 0,
                      channel: [str, int] = 0,
                      key: str = None,
                      *args, **kwargs
                      ) -> np.ndarray:
        """
        I want this function to be able to generate arbitrary
        masks from some default functions
        i.e. percentiles, abs_val, etc..
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
                mask = getattr(filt, function)(vals, *args, **kwargs)
            except AttributeError:
                raise AttributeError('Did not understand filtering '
                                     f'function {function}.')

        if key is not None:
            # Save the mask if key is given
            self._nan_mask[key] = mask

        return mask

    def get_mask(self, key: str) -> np.ndarray:
        return self._nan_mask[key]


class PositionArray():
    """
    Add Typing hints when the imports are fixed

    TODO:
        - Add ability to filter all CellArrays
    """

    __slots__ = ('name', 'attrs', 'sites', '_all_masks')

    def __init__(self,
                 arrays: Collection[CellArray] = None,
                 name: str = None,
                 attrs: dict = None,
                 **kwargs
                 ) -> None:
        """
        TODO:
            - Will input dimensions ever need to be padded?
        """
        # Save some values
        self.name = name
        self.attrs = attrs
        self.sites = {}

        # Save arrays if given
        if arrays is not None:
            [self.__setitem__(None, a) for a in arrays]

        # Build dictionary for saving masks
        self._all_masks = {k: {} for k in self.sites}

    def __setitem__(self, key=None, value=None):

        # If no value is passed, nothing is done
        if value is None:
            return
        elif not isinstance(value, CellArray):
            raise TypeError('All values in PositionArray must be'
                            f'type CellArray. Got {type(value)}.')

        # Get key from array or set increment
        if key is None:
            key = len(self.sites) + 1 if value.name is None else value.name

        self.sites[key] = value

    def __getitem__(self, key):
        """
        Assumption is that all Arrays have the same coordinates.
        The first Array is used to generate the dictionaries for indexing
        """
        try:
            # First try to return a site the user requested
            return self.sites[key]
        except KeyError:
            # If site doesn't exist, try using key to index all sites
            if not isinstance(key, tuple):
                key = tuple([key])
            indices = tuple(self.sites.values())[0]._convert_keys_to_index(key)

            # TODO: Should this return a dictionary or list/tuple?
            return {k: v._getitem_w_idx(indices)
                    for k, v in self.sites.items()}
            # return [v._getitem_w_idx(indices) for v in self.sites.values()]

    def __len__(self):
        return len(self.sites)

    def __str__(self):
        return str(self.sites)

    @property
    def shape(self):
        return {k: v.shape for k, v in self.sites.items()}

    @property
    def ndim(self):
        return {k: v.ndim for k, v in self.sites.items()}

    @property
    def dtype(self):
        return {k: v.dtype for k, v in self.sites.items()}

    def save(self, path: str) -> None:
        """
        Saves all the CellArrays in PositionArray
        to an hdf5 file.

        TODO:
            - Add checking for path and overwrite options
        """
        f = h5py.File(path, 'w')
        for key, val in self.sites.items():
            # Array data stored as a dataset
            f.create_dataset(key, data=val._arr)

            # Axis names and coords stored as attributes
            for coord in val.coords:
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

    def generate_mask(self,
                      function: [Callable, str],
                      metric: str,
                      region: [str, int] = 0,
                      channel: [str, int] = 0,
                      key: str = None,
                      *args, **kwargs
                      ) -> np.ndarray:
        """
        Calls generate_mask for each CellArray in self.sites
        """
        # Build masks in each CellArray with the function
        _masks = [v.generate_mask(function, metric, region,
                                  channel, key, *args, **kwargs)
                  for v in self.sites.values()]

        # Save if needed
        if key is not None:
            self._all_masks[key] = _masks

        return _masks

    def filter_cells(self,
                     mask: Collection[np.ndarray] = None,
                     key: str = None,
                     delete: bool = True,
                     *args, **kwargs
                     ) -> np.ndarray:
        """
        Either uses an arbitrary mask or a saved mask (key) to
        filter the data. If delete, the underlying structure
        is changed, otherwise, the data are only returned.

        TODO:
            - Add option to return a new CellArray instead of
              an np.ndarray
        """
        # If key is provided, look for saved mask
        if mask is None and key is not None:
            mask = self._all_masks[key]
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

        # Mask type and dimension will be checked in CellArray
        out = {site: arr.filter_cells(msk, key, delete, *args, **kwargs)
               for msk, (site, arr) in zip(mask, self.sites.items())}

        return out

    @staticmethod
    def _build_from_file(f: h5py.File) -> 'PositionArray':
        """
        Given an hdf5 file, returns a PositionArray instance
        """
        pos = PositionArray()
        for key in f:
            # Attrs define the coords and axes
            _arr = CellArray(**f[key].attrs, name=key)
            # f[key] holds the actual array data
            _arr[:] = f[key]
            pos[key] = _arr

        return pos


# Define custom types to make output tracking esier
Image = NewType('image', np.ndarray)
Mask = NewType('mask', np.ndarray)
Track = NewType('track', np.ndarray)
Arr = NewType('array', CellArray)

# Save input names and types
INPT_NAMES = [Image.__name__, Mask.__name__, Track.__name__, Arr.__name__]
INPT_NAME_IDX = {n: i for i, n in enumerate(INPT_NAMES)}
INPT = [Image, Mask, Track, Arr]
INPT_IDX = {n: i for i, n in enumerate(INPT)}
