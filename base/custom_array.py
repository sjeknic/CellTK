import xarray as xr
import numpy as np


class CustomArray(xr.DataArray):
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
    __name__ = 'customarray'
    __slots__ = ()

    def __init__(self,
                 arr: np.ndarray,
                 coords: dict = None,
                 name: str = None,
                 attrs: dict = None,
                 **kwargs
                 ) -> None:
        """
        dims are constant, so they aren't included as an input arg
        """
        dims = ['region', 'channel', 'metric', 'cell', 'frames']
        super().__init__(data=arr, coords=coords, dims=dims,
                         name=name, attrs=attrs)
