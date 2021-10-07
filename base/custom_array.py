from typing import Tuple, Collection

import xarray as xr
import numpy as np


class CustomArray():
    """
    ax 0 - cell locations (nuc, cyto, population, etc.)
    ax 1 - channels (TRITC, FITC, etc.)
    ax 2 - metrics (median_int, etc.)
    ax 3 - cells
    ax 4 - frames

    Question: what axes are the fastest and slowest to write too. Does that count still
    if I'm using xarr. And do I even need xarr still, because I am skipping most of the
    actual functions

    """
    __slots__ = ('_xarr', 'name', 'attrs', 'coords', '_arr_dim', '_dim_idxs',
                 '_key_dim_pairs', '_key_coord_pairs', '_nan_mask')

    def __init__(self,
                 regions: Collection[str] = ['nuc'],
                 channels: Collection[str] = ['TRITC'],
                 metrics: Collection[str] = ['label'],
                 cells: Collection[int] = [0],
                 frames: Collection[int] = [0],
                 name: str = None,
                 attrs: dict = None,
                 **kwargs
                 ) -> None:
        """
        dims are constant, so they aren't included as an input arg

        TODO:
            - Indexing by str with xr.DataArray is very slow. Might want to
              move to a system based only on np.ndarray. The only question now
              is relating to the DataSet structure when all of the np arrays
              are implemented. The simplest way might just be to index all of them
              and stack the results. But for now keeping as is
            - Handling of metrics which will have multiple entries (i.e. bbox-0, bbox-1, etc.)
              The simplest might be to just automatically convert the keys when the array is loaded.
              The issue with this is that this would also have to happen in __getitem__, which
              seems like it would slow things down. One option could be to check if any of those metrics
              exist when the CustomArray is made. If they are __getitem__ first has to call a function to
              sort that out, and if not, that function could just be a pass through function.
            - Reorder if-statements for speed in key_coord functions.
            - Add option to return nans, for now default is to not return them
            - Add ability to save time steps
        """
        # Save some values
        self.name = name
        self.attrs = attrs

        # Set _arr_dim based on input values
        self._arr_dim = (len(regions), len(channels), len(metrics),
                         len(cells), len(frames))
        # Create empty data array
        arr = np.empty(self._arr_dim)

        # Create coordinate dictionary
        self.coords = dict(region=regions, channel=channels, metric=metrics,
                           cell=cells, frame=frames)
        self._make_key_coord_pairs(self.coords)

        self._xarr = xr.DataArray(data=arr, coords=self.coords,
                                  name=name, attrs=attrs)
        self._nan_mask = np.empty(self._xarr.values.shape).astype(bool)
        self._nan_mask[:] = True

    @property
    def shape(self):
        return self._xarr.shape

    @property
    def ndim(self):
        return self._xarr.ndim

    def __getitem__(self, key):
        # Needed if only one key is passed
        if not isinstance(key, tuple):
            key = tuple([key])
        # Sort given indices to the appropriate axes
        indices = self._convert_keys_to_index(key)

        # Always return as a numpy array
        return self._xarr.values[indices]

    def __setitem__(self, key, value):
        # Sort given indices to the appropriate axes
        if not isinstance(key, tuple):
            key = tuple([key])
        indices = self._convert_keys_to_index(key)

        self._xarr.values[indices] = value

    def _getitem_w_idx(self, idx):
        """
        Used by CustomSet to index CustomArray w/o recalculating
        the indices each time
        """
        return self._xarr.values[idx]

    def __str__(self):
        return self._xarr.__str__()

    def _convert_keys_to_index(self, key) -> Tuple[(int, slice)]:
        """
        Converts strings and slices given in key to the saved
        dimensions and coordinates of self._xarr.

        Args:
            - key: tuple(slices)

        Returns:
            tuple(slices)

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
        cell_idx = self._dim_idxs['cell']
        frame_idx = self._dim_idxs['frame']
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
            else:
                raise ValueError(f'Indices must be int, str, or slice. Got {type(k)}.')

        return tuple(indices)

    def _make_key_coord_pairs(self, coords: dict) -> None:
        """
        """
        # Convert string axis index to integer index
        self._dim_idxs = {k: n for n, k in enumerate(coords.keys())}

        # Check for duplicate keys (allowed in cell and frame)
        to_check = ['region', 'channel', 'metric']
        all_poss = [sl for l in [coords[t] for t in to_check] for sl in l]
        if len(all_poss) != len(set(all_poss)):
            raise KeyError(f'All coordinates in dimensions {to_check} must be '
                           f'unique. Got {all_poss}.')

        # Match keys and coordinates
        self._key_dim_pairs = {
            a: [k for k, v in coords.items() if a in tuple(v)][0] for a in all_poss
        }
        # NOTE: Not sure I will need this one, could just use xarr internals
        self._key_coord_pairs = {
            a: [v.index(a) for k, v in coords.items() if a in tuple(v)][0] for a in all_poss
        }


class CellSet():
    """
    Add Typing hints when the imports are fixed
    """

    __slots__ = ('name', 'attrs', 'sites')

    def __init__(self,
                 arrays: Collection = None,
                 name: str = None,
                 attrs: dict = None,
                 **kwargs
                 ) -> None:
        """
        """
        # Save some values
        self.name = name
        self.attrs = attrs
        self.sites = {}

        if arrays is not None:
            [self.__setitem__(None, a) for a in arrays]

    def __setitem__(self, key=None, value=None):

        # If no value is passed, nothing is done
        if value is None:
            return

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

            return [v._getitem_w_idx(indices) for v in self.sites.values()]

    def __len__(self):
        return len(self.sites)

    def __str__(self):
        return str(self.sites)
