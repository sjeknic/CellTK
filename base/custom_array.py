from typing import Tuple, Collection

import xarray as xr
import numpy as np


class CustomArray():
    """
    This will hold the data for one site (hopefully)
         The question will probably be exactly how and what dimensions will line up...

        * numbers are arbitrary, will change based on read/write speed.
        Opt 2:
            ax 0 - cell locations (nuc, cyto, population, etc.)
            ax 1 - channels (TRITC, FITC, etc.)
            ax 2 - metrics (median_int, etc.)
            ax 3 - cells
            ax 4 - frames

    Another question is how will this CustomArray actually be built. I think that
    depends a lot on how the data extraction looks like. So I'm going to start thinking
    about that as well. For now, the whole array has to be provided....
    """
    __slots__ = ('xarr', 'name', 'attrs', 'coords', '_dim_idxs',
                 '_key_dim_pairs', '_key_coord_pairs')

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
        """
        # Save some values
        self.name = name
        self.attrs = attrs

        # Set _arr_dim based on input values
        _arr_dim = (len(regions), len(channels), len(metrics),
                    len(cells), len(frames))
        # Create empty data array
        arr = np.empty(_arr_dim)

        # Create coordinate dictionary
        self.coords = dict(region=regions, channel=channels, metric=metrics,
                           cell=cells, frame=frames)
        self._make_key_coord_pairs(self.coords)

        self.xarr = xr.DataArray(data=arr, coords=self.coords,
                                 name=name, attrs=attrs)

    @property
    def shape(self):
        return self.xarr.shape

    def __getitem__(self, key):
        """
        """
        # Sort given indices to the appropriate axes
        if not isinstance(key, tuple):
            key = tuple([key])
        indices = self._convert_keys_to_index(key)

        # Always return as a numpy array
        return self.xarr.values[indices]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = tuple([key])
        indices = self._convert_keys_to_index(key)

        self.xarr.values[indices] = value

    def __str__(self):
        return self.xarr.__str__()

    def _convert_keys_to_index(self, key):
        """
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
                    indices[k_idx] = k
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
                        raise KeyError(f"Dimensions don't match: {k.start} is in "
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
            a: [k for k, v in coords.items() if a in v][0] for a in all_poss
        }
        # NOTE: Not sure I will need this one, could just use xarr internals
        self._key_coord_pairs = {
            a: [v.index(a) for k, v in coords.items() if a in v][0] for a in all_poss
        }
