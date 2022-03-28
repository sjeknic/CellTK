from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from celltk.utils.unet_model import UPeakModel
from celltk.utils.plot_utils import plot_trace_predictions


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
                  weight_path: str = 'celltk/config/upeak_example_weights.tf',
                  segment: bool = True,
                  roi: Union[int, Tuple[int]] = (1, 2),
                  save_path: str = None,
                  save_plot: str = None,
                  show_plot: bool = False
                  ) -> None:
    """Predicts peaks in arbitrary data using UPeak.

    :param array: Data of shape n_samples x n_timepoints. Each sample will
        have peaks predicted.
    :param weight_path: Weights to use for predicting peaks. If not provided,
        uses default weights.
    :param segment: If True, segments peaks using a watershed based algorithm
        and returns segmennted peaks instead of peak probabilities.
    :param roi: Regions of interest to return. 0 is background, 1 is slope, and
        2 is plateau.
    :param save_path: If provided, saves the output array at the supplied path.
    :param save_plot: If a string is provided, saves output figures using the given
        figure name. Do not provided extension in file name, figures can currently
        only be saved as png.
    :param show_plot:
    """
    # Get model
    model = UPeakModel(weight_path)
    predictions = model.predict(array, roi=roi)

    if segment:
        # This will overwrite predictions, the segmentation is returned/plotted
        pass

    # Save the outputs
    if save_path:
        np.save(save_path, predictions)

    if save_plot or show_plot:
        for fidx, fig in enumerate(plot_trace_predictions(array, predictions)):
            if save_plot: plt.savefig(fig, f'{save_plot}_{fidx}.png')
            if show_plot: plt.show()
            plt.close(fig)

    return predictions
