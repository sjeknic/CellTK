from typing import List, Collection

import numpy as np
import skimage.segmentation as segm
import scipy.integrate as integrate

from celltk.utils.filter_utils import outside, inside

def segment_peaks_agglomeration(traces: np.ndarray,
                                probabilities: np.ndarray,
                                steps: int = 15,
                                min_seed_prob: float = 0.8,
                                min_peak_prob: float = 0.5,
                                min_seed_length: int = 2,
                                **kwargs  # Messy fix for running this from derived metrics
                                ) -> np.ndarray:
    """Returns an array with peaks incrementally counted in each trace

    I think I want to just track peaks with a label/mask.
    As in the labels will be [0, 0, 1, 1,...0, 2, 2, ... 0, 3 ..]
    And the mask can just be labels > 0
    That should work for everything...

    0 - BG, 1 - slope, 2 - plateau

    TODO:
        - Add option for user-passed seeds
    """
    # Make sure traces and probabilities match
    assert traces.shape[:2] == probabilities.shape[:2]

    # Probabilities should be 3D. If 2D, assume slope + plateau
    assert probabilities.ndim == 3
    if probabilities.shape[-1] == 3:
        # Background probability is not needed
        probabilities = probabilities[..., 1:]
    elif probabilities.shape[-1] < 2 or probabilities.shape[-1] > 3:
        raise ValueError('Expected 2 or 3 classes in probabilities. '
                         f'Got {probabilities.shape[-1]}.')

    # Extract individual probabilities
    slope, plateau = (probabilities[..., 0], probabilities[..., 1])
    # Apply to each trace
    out = np.zeros(traces.shape, dtype=np.uint8)
    for n, (t, s, p) in enumerate(zip(traces, slope, plateau)):
        out[n] = _peak_labeler(t, s, p)

    return out


def _peak_labeler(trace: np.ndarray,
                  slope: np.ndarray,
                  plateau: np.ndarray,
                  steps: int = 15,
                  min_seed_prob: float = 0.8,
                  min_peak_prob: float = 0.5,
                  min_seed_length: int = 2
                  ) -> np.ndarray:
    """Gets 1D trace and returns with peaks labeled
    """
    # Get seeds based on constant probability
    seeds = _idxs_to_labels(
        trace, _constant_thres_peaks(plateau, min_seed_prob, min_seed_length)
    )

    # Use iterative watershed to segment
    peaks = _agglom_watershed_peaks(trace, seeds, slope + plateau,
                                    steps, min_peak_prob)

    return peaks


def _constant_thres_peaks(probability: np.ndarray,
                          min_probability: float = 0.8,
                          min_length: int = 8,
                          max_gap: int = 2
                          ) -> List[np.ndarray]:
    """"""
    candidate_pts = np.where(probability >= min_probability)[0]

    # Find distances between candidates
    diffs = np.ediff1d(candidate_pts, to_begin=1)
    bounds = np.where(diffs >= max_gap)[0]

    return [p for p in np.split(candidate_pts, bounds) if len(p) >= min_length]


def _agglom_watershed_peaks(trace: np.ndarray,
                            seeds: np.ndarray,
                            probability: np.ndarray,
                            steps: int = 15,
                            min_probability: float = 0.5
                            ) -> np.ndarray:
    """
    watershed is based on trace value, not peak probability
    """
    out = np.zeros_like(seeds)
    if seeds.any():
        # Global mask for all steps
        cand_mask = probability >= min_probability

        # Iterate through all of the steps
        perclist = np.linspace(np.nanmax(trace), np.nanmin(trace), steps)
        _old_perc = perclist[0]
        for _perc in perclist:
            # Get the mask for this step
            mask = np.logical_and(trace > _perc, trace <= _old_perc)
            # Seeds are always included, cand_mask always matters
            mask = np.logical_or(seeds > 0, (mask * cand_mask) > 0)

            # Watershed and save the seeds for the next itreation
            # TODO: Is compactness actually needed?
            seeds = segm.watershed(trace, markers=seeds, mask=mask,
                                   watershed_line=True, compactness=5)
            out = seeds

    return out


def _idxs_to_labels(trace: np.ndarray, indexes: List[np.ndarray]) -> np.ndarray:
    """"""
    out = np.zeros(trace.shape, dtype=np.uint8)
    for label, pts in enumerate(indexes):
        out[pts] = label + 1

    return out


def _labels_to_idxs(labels: np.ndarray) -> List[np.ndarray]:
    """"""
    if labels.ndim == 1: labels = labels[None, :]
    out = []
    for lab in labels:
        lab = np.unique(lab[lab > 0])
        out.append([np.where(labels == l)[0] for l in lab])

    return out
class PeakMetrics():
    """Helper class for working with output peaks"""

    # Prominence/Height
    def amplitude(self,
                  traces: np.ndarray,
                  labels: np.ndarray
                  ) -> np.ndarray:
        """"""
        # TODO: Rewrite to be like the old one
        out = np.zeros_like(traces)
        out = []
        for trace, label in zip(traces, labels):
            _temp = []
            for l in np.unique(label[label > 0]):
                mask = np.where(label == l, trace, 0)
                _temp.append(np.max(mask))
            out.append(_temp)
        return out

    # Width
    def length(self,
               traces: np.ndarray,
               labels: np.ndarray
               ) -> List[List[int]]:
        """Total peak length"""
        out = []
        for lab in labels:
            peak, counts = np.unique(lab[lab > 0], return_counts=True)
            out.append(list(counts))

        return out
    # Miscellaneous
    def detect_peak_tracts(self,
                           traces: np.ndarray,  # Not used in this function
                           labels: np.ndarray,
                           max_gap: int = 8
                           ) -> np.ndarray:
        """Connects peaks that are close together into a single tract.
        """
        # out = np.zeros_like(labels)
        out = []
        for idx, lab in enumerate(labels):
            if lab.any():
                # Find labels separated by greater than max gap
                p_idx = np.where(lab > 0)[0]
                diffs = np.ediff1d(p_idx, to_begin=1)
                bounds = np.where(diffs > max_gap)[0]
                # Sort into unique tracts
                out.append([np.unique(t)
                            for t in np.split(lab[p_idx], bounds)])
            else:
                out.append([])

        return out

    def filter_peaks(self,
                     traces: np.ndarray,
                     labels: np.ndarray,
                     metrics: Collection[str],
                     thresholds: Collection[float]
                     ) -> np.ndarray:
        """"""
        mask = np.ones(labels.shape, dtype=bool)
        for metric, thres in zip(metrics, thresholds):
            data = getattr(self, metric)(traces, labels)
            data_array = self._results_to_array(traces, labels, data)
            # outside returns array where points "to keep" are True
            # So use inverse of the mask
            mask *= outside(data_array, lo=thres, propagate=False)

        # delete the peaks
        labels[~mask] = 0
        return labels

    @staticmethod
    def _results_to_array(traces: np.ndarray,
                          labels: np.ndarray,
                          results: List[List[float]]
                          ) -> np.ndarray:
        """This inputs the traces and results [[cell1]...[cellN]]
        and returns array same shape as traces with the indices of the
        peak overwritten with the results for that peak. All other indices are zero"""
        out = np.zeros_like(traces)
        for n, (label, result) in enumerate(zip(labels, results)):
            for peak, r in enumerate(result):
                peak += 1  # peaks are 1-indexed
                out[n, label == peak] = r
        return out
