import numpy as np


def median_intensity(mask, image) -> float:
    """
    Returns median intensity in region of interest
    """
    return np.median(image[mask])
