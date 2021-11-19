import numpy as np

"""For all, if values is 2D, assume cells x frames
and apply the mask only in first axis"""


def _propagate_mask(mask: np.ndarray) -> np.ndarray:
    """
    For any cell with a single value masked, mask all values
    """
    if mask.ndim == 2:
        # If a single False is in a row, mark all False
        mask[(~mask).any(1), :] = False

    return mask


def outside(values: np.ndarray,
            lo: float = -np.inf,
            hi: float = np.inf,
            allow_equal: bool = True
            ) -> np.ndarray:
    """
    Masks cells with values that fall outside of the specified range
    """
    # Set values that are within the range to True
    if allow_equal:
        ma = np.logical_and(values >= lo, values <= hi)
    else:
        ma = np.logical_and(values > lo, values < hi)

    return _propagate_mask(ma)


def inside(values: np.ndarray,
           lo: float = -np.inf,
           hi: float = np.inf,
           allow_equal: bool = True
           ) -> np.ndarray:
    """
    Masks cells with values that fall inside of the specified range
    """
    return ~outside(values, lo, hi)


def outside_percentile(values: np.ndarray,
                       lo: float = 5,
                       hi: float = 95,
                       allow_equal: bool = True
                       ) -> np.ndarray:
    """
    Masks cells with values that fall outside of the specified
    percentile range
    """
    # Compute values of the boundries
    lo = np.percentile(values, lo)
    hi = np.percentile(values, hi)

    return outside(values, lo, hi)


def inside_percentile(values: np.ndarray,
                      lo: float = 5,
                      hi: float = 95,
                      allow_equal: bool = True
                      ) -> np.ndarray:
    """
    Masks cells with values that fall inside of the specified
    percentile range
    """
    return ~outside_percentile(values, lo, hi)


def any_nan(values: np.ndarray) -> np.ndarray:
    """
    Masks cells that include any np.nan
    """
    return _propagate_mask(np.isnan(values))


def any_negative(values: np.ndarray) -> np.ndarray:
    """
    Masks cells that include any negative values
    """
    return _propagate_mask(values < 0)
