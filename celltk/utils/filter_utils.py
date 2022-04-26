import numpy as np

"""
Cells are to be removed are marked with False.

For all, if values is 2D, assume cells x frames
and apply the mask only in first axis.

TODO: Add 1D filters to this file as well
"""


def _propagate_mask(mask: np.ndarray) -> np.ndarray:
    """
    For any cell with a single value masked, mask all values
    """
    if mask.ndim == 2:
        # If a single False is in a row, mark all False
        mask[np.logical_not(mask).any(1), :] = False

    return mask


def _clean_mask(values: np.ndarray,
                mask: np.ndarray,
                ignore_nans: bool,
                propagate: bool,
                user_mask: np.ndarray = None
                ) -> np.ndarray:
    """"""
    # Set nans to True to exclude from masking
    if ignore_nans:
        mask[np.isnan(values)] = True

    # user_mask marks frames to be excluded from filtering with True
    if user_mask is not None:
        mask[user_mask] = True

    # A single False will exclude the whole row
    if propagate:
        mask = _propagate_mask(mask)

    return mask


def outside(values: np.ndarray,
            lo: float = -np.inf,
            hi: float = np.inf,
            allow_equal: bool = True,
            ignore_nans: bool = False,
            propagate: bool = True,
            mask: np.ndarray = None,
            ) -> np.ndarray:
    """
    Masks cells with values that fall outside of the specified range
    """
    # Set values that are within the range to True
    if allow_equal:
        ma = np.logical_and(values >= lo, values <= hi)
    else:
        ma = np.logical_and(values > lo, values < hi)
    return _clean_mask(values, ma, ignore_nans, propagate, mask)


def inside(values: np.ndarray,
           lo: float = -np.inf,
           hi: float = np.inf,
           allow_equal: bool = True,
           ignore_nans: bool = False,
           propagate: bool = True,
           mask: np.ndarray = None,
           ) -> np.ndarray:
    """
    Masks cells with values that fall inside of the specified range
    """
    # Set values that are outside the range to True
    if allow_equal:
        ma = np.logical_or(values <= lo, values >= hi)
    else:
        ma = np.logical_or(values < lo, values > hi)

    return _clean_mask(values, ma, ignore_nans, propagate, mask)


def outside_percentile(values: np.ndarray,
                       lo: float = 0,
                       hi: float = 100,
                       ignore_nans: bool = False,
                       propagate: bool = True,
                       mask: np.ndarray = None,
                       ) -> np.ndarray:
    """
    Masks cells with values that fall outside of the specified
    percentile range
    """
    # Compute values of the boundries
    if ignore_nans:
        lo = np.nanpercentile(values, lo)
        hi = np.nanpercentile(values, hi)
    else:
        lo = np.percentile(values, lo)
        hi = np.percentile(values, hi)
    ma = outside(values, lo, hi, propagate=False)

    return _clean_mask(values, ma, ignore_nans, propagate, mask)


def inside_percentile(values: np.ndarray,
                      lo: float = 0,
                      hi: float = 100,
                      ignore_nans: bool = False,
                      propagate: bool = True,
                      mask: np.ndarray = None,
                      ) -> np.ndarray:
    """
    Masks cells with values that fall inside of the specified
    percentile range
    """
    # Compute values of the boundries
    if ignore_nans:
        lo = np.nanpercentile(values, lo)
        hi = np.nanpercentile(values, hi)
    else:
        lo = np.percentile(values, lo)
        hi = np.percentile(values, hi)
    ma = inside(values, lo, hi, propagate=False)

    return _clean_mask(values, ma, ignore_nans, propagate, mask)


def any_nan(values: np.ndarray,
            propagate: bool = True,
            mask: np.ndarray = None,
            ) -> np.ndarray:
    """
    Masks cells that include any np.nan
    """
    ma = ~np.isnan(values)

    # Obviously ignore_nans must be false for this mask
    return _clean_mask(values, ma, False, propagate, mask)


def any_negative(values: np.ndarray,
                 ignore_nans: bool = False,
                 propagate: bool = True,
                 mask: np.ndarray = None
                 ) -> np.ndarray:
    """
    Masks cells that include any negative values
    """
    ma = ~(values < 0)

    return _clean_mask(values, ma, ignore_nans, propagate, mask)