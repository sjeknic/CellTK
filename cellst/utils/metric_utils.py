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
