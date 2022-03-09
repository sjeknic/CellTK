import numpy as np

'''TODO: Add nan policy'''


def median_intensity(mask, image) -> float:
    """Returns median intensity in region of interest."""
    return np.median(image[mask])


def total_intensity(mask, image) -> float:
    """Returns total intensity in region of interest."""
    return np.sum(image[mask])


def intensity_variance(mask, image) -> float:
    """Returns variance of all intensity values in region of interest."""
    return np.var(image[mask])


def intensity_stdev(mask, image) -> float:
    """
    Returns standard deviation of all intensity values in region of interest.
    """
    return np.std(image[mask])


# VVV These are more useful as derived_metrics VVV #
def active_cells(array: np.ndarray,
                 thres: np.ndarray
                 ) -> np.ndarray:
    """"""
    return array >= thres

def cumulative_active(array: np.ndarray) -> np.ndarray:
    """"""
    return np.where(np.cumsum(array, axis=1), 1, 0)
