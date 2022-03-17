import warnings

import numpy as np


def median_intensity(mask: np.ndarray, image: np.ndarray) -> float:
    """Returns median intensity in region of interest."""
    return np.median(image[mask])


def total_intensity(mask: np.ndarray, image: np.ndarray) -> float:
    """Returns total intensity in region of interest."""
    return np.sum(image[mask])


def intensity_variance(mask: np.ndarray, image: np.ndarray) -> float:
    """Returns variance of all intensity values in region of interest."""
    return np.var(image[mask])


def intensity_stdev(mask: np.ndarray, image: np.ndarray) -> float:
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


def predict_peaks(array: np.ndarray,
                  weight_path: str = 'celltk/config/example_peak_weights.tf',
                  segment: bool = True,
                  ) -> None:
    """Placeholder function for now, but can change to include peaak prediction"""
    pass