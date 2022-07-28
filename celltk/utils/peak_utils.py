from itertools import zip_longest
from typing import List, Collection, Tuple

import numpy as np
import skimage.segmentation as segm
import scipy.integrate as integrate
import scipy.stats as stats

from celltk.utils.filter_utils import outside, inside

def segment_peaks_agglomeration(traces: np.ndarray,
                                probabilities: np.ndarray,
                                steps: int = 15,
                                min_seed_prob: float = 0.9,
                                min_peak_prob: float = 0.3,
                                min_seed_length: int = 2,
                                **kwargs  # Messy fix for running this from derived metrics
                                ) -> np.ndarray:
    """Returns an array with peaks incrementally counted in each trace,
    i.e. the labels will be [0, 0, 1, 1,...0, 2, 2, ... 0, 3 ..].

    TODO:
        - Add option for user-passed seeds
    """
    # Make sure traces and probabilities match
    assert traces.shape[:2] == probabilities.shape[:2]

    # Apply to each trace
    out = np.zeros(traces.shape, dtype=np.uint8)
    for n, (t, p) in enumerate(zip(traces, probabilities)):
        out[n] = _peak_labeler(t, p)

    return out


def _peak_labeler(trace: np.ndarray,
                  probability: np.ndarray,
                  steps: int = 15,
                  min_seed_prob: float = 0.9,
                  min_peak_prob: float = 0.3,
                  min_seed_length: int = 2
                  ) -> np.ndarray:
    """Gets 1D trace and returns with peaks labeled
    """
    # Get seeds based on constant probability
    seeds = _idxs_to_labels(
        trace, _constant_thres_peaks(probability, min_seed_prob,
                                     min_seed_length)
    )

    # Use iterative watershed to segment
    peaks = _agglom_watershed_peaks(trace, seeds, probability,
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


def _idxs_to_labels(trace: np.ndarray,
                    indexes: List[np.ndarray]
                    ) -> np.ndarray:
    """"""
    # Takes points associated with peak and labels a 1D ndarray
    out = np.zeros(trace.shape, dtype=np.uint8)
    for label, pts in enumerate(indexes):
        out[pts] = label + 1

    return out


def _labels_to_idxs(labels: np.ndarray) -> List[np.ndarray]:
    """Idxs """
    if labels.ndim == 1: labels = labels[None, :]
    out = []
    for lab in labels:
        peaks = np.unique(lab[lab > 0])
        out.append([np.unique(np.where(lab == p)[0]) for p in peaks])

    return out


class PeakHelper:
    """Helper class for getting data from traces and peak labels"""
    def first_time(self,
                   traces: np.ndarray,
                   labels: np.ndarray,
                   ) -> List[List[float]]:
        """Returns the first time point belonging to each peak"""
        idxs = _labels_to_idxs(labels)
        out = []
        for trace, idx in zip(traces, idxs):
            out.append(self._times(trace, idx, 'first'))
        return out

    def last_time(self,
                  traces: np.ndarray,
                  labels: np.ndarray
                  ) -> List[List[float]]:
        """Returns the last time point belonging to each peak"""
        idxs = _labels_to_idxs(labels)
        out = []
        for trace, idx in zip(traces, idxs):
            out.append(self._times(trace, idx, 'last'))
        return out

    def max_time(self,
                 traces: np.ndarray,
                 labels: np.ndarray
                 ) -> List[List[float]]:
        """Returns the time that a peak reaches it's maximum amplitude."""
        idxs = _labels_to_idxs(labels)
        out = []
        for trace, idx in zip(traces, idxs):
            out.append(self._times(trace, idx, 'max'))
        return out

    def amplitude(self,
                  traces: np.ndarray,
                  labels: np.ndarray
                  ) -> List[List[float]]:
        """Returns the maximum value in each peak for each trace.
        Each cell has an empty list if no peaks are labeled.

        :param traces: Array of shape n_cells x n_features with values
            to use for calculating amplitude.
        :param labels: Array of same shape as traces with peaks labeled
            with unique integers in each cell trace.

        :return: List of amplitudes for each cell trace.
        """
        out = []
        for trace, label in zip(traces, labels):
            out.append(self._amplitude(trace, label))
        return out

    def prominence(self,
                   traces: np.ndarray,
                   labels: np.ndarray,
                   tracts: List[List[int]] = []
                   ) -> List[List[float]]:
        """Returns the difference between the maximum value in
        a peak and the base of the peak. If tracts are provided
        adjusts the base of each peak in the tract to be the base
        of the tract.

        :param traces:
        :param labels:
        :param tracts:

        :return:
        """
        idxs = _labels_to_idxs(labels)
        amps = self.amplitude(traces, labels)

        out = []
        long_zip = zip_longest(traces, idxs, amps, tracts, fillvalue=[])
        for trace, idx, amp, tract in long_zip:
            out.append(self._prominence(trace, idx, amp, tract))
        return out

    def length(self,
               traces: np.ndarray,
               labels: np.ndarray
               ) -> List[List[int]]:
        """Returns the length of each peak.

        :param traces:
        :param labels:

        :return:
        """
        out = []
        for lab in labels:
            out.append(self._length(lab))

        return out

    def width(self,
              traces: np.ndarray,
              labels: np.ndarray,
              tracts: List[List[int]] = [],
              relative: float = 0.5,
              absolute: float = None
              ) -> List[List[float]]:
        """Not yet complete.

        :param traces:
        :param labels:
        :param tracts:
        :param relative:
        :param absolute:

        :return:
        """
        idxs = _labels_to_idxs(labels)

        if not tracts:
            tracts = self.detect_peak_tracts(traces, labels)
        amps = self.amplitude(traces, labels)
        proms = self.prominence(traces, labels, tracts)

        out = []
        long_zip = zip_longest(traces, idxs, tracts, amps, proms,
                               fillvalue=[])
        for trace, idx, tract, amp, prom in long_zip:
            out.append(self._width(trace, idx, tract, amp, prom,
                                   relative, absolute))
        return out

    def nonlinearity(self,
                     traces: np.ndarray,
                     labels: np.ndarray,
                     ) -> List[List[float]]:
        """Returns inverse of the absolute value of the
        Pearson's correlation coefficient. Higher values
        mean the peak is less linear.

        :param traces:
        :param labels:

        :return:
        """
        out = []
        for trace, label in zip(traces, labels):
            out.append(self._linearity(trace, label))

        return out

    def area_under_curve(self,
                         traces: np.ndarray,
                         labels: np.ndarray
                         ) -> List[List[float]]:
        """
        TODO:
            - Add a way to specify t or dt
        """
        out = []
        for trace, label in zip(traces, labels):
            out.append(self._area_under_curve(trace, label))
        return out

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
                out.append([np.unique(t).astype(np.uint8)
                            for t in np.split(lab[p_idx], bounds)])
            else:
                out.append([])

        return out

    def filter_peaks(self,
                     traces: np.ndarray,
                     labels: np.ndarray,
                     metrics: Collection[str],
                     thresholds: Collection[float],
                     kwargs: Collection[dict] = [{}]
                     ) -> np.ndarray:
        """
        TODO:
            - Add both high and low thresholds
        """
        mask = np.ones(labels.shape, dtype=bool)
        for metric, thres, kws in zip_longest(metrics,
                                              thresholds,
                                              kwargs,
                                              fillvalue={}):
            data = getattr(self, metric)(traces, labels, **kws)
            data_array = self._results_to_array(traces, labels, data)
            # outside returns array where points "to keep" are True
            # So use inverse of the mask to remove other points
            mask *= outside(data_array, lo=thres, propagate=False)

        # delete the peaks
        labels = labels.copy()
        labels[~mask] = 0

        # # Relabel peaks to be sequential
        for i, lab in enumerate(labels):
            _lab = np.unique(lab[lab > 0])
            if len(_lab) > 0:
                for n, l in enumerate(_lab):
                    n += 1  # peaks are 1-indexed
                    labels[i, lab == l] = n

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

    @staticmethod
    def _times(trace: np.ndarray,
               index: np.ndarray,
               time: str,
               ) -> List[float]:
        """
        TODO:
            - This will not work for traces containing all
              negative values.
            - Add a min option
        """
        if time == 'first':
            return [idx[0] for idx in index]
        elif time == 'last':
            return [idx[-1] for idx in index]
        elif time == 'max':
            out = []
            for idx in index:
                _tr = np.zeros_like(trace)
                _tr[idx] = trace[idx]
                out.append(np.argmax(trace[idx]))
            return out

    @staticmethod
    def _amplitude(trace: np.ndarray,
                   label: np.ndarray,
                   ) -> List[float]:
        """"""
        out = []
        for l in np.unique(label[label > 0]):
            mask = np.where(label == l, trace, 0)
            out.append(np.nanmax(mask))

        return out

    @staticmethod
    def _prominence(trace: np.ndarray,
                    index: np.ndarray,
                    amplitude: List[float],
                    tract: List[List[int]]
                    ) -> List[float]:
        """"""
        _edge_dist = 4
        out = []
        if tract:
            for t in tract:
                # Peaks are 1-indexed
                frst_pk = t[0] - 1
                last_pk = t[-1] - 1

                # Get first and last point in tract
                x = [index[frst_pk][0], index[last_pk][-1]]
                y = [trace[x[0]], trace[x[-1]]]

                # Adjust heights if close to edge
                if abs(x[0] - 0) <= _edge_dist:
                    y[0] = y[-1]
                if abs(x[1] - len(trace) - 1) <= _edge_dist:
                    y[-1] = y[0]
                _base = np.mean(y)

                # For each peak in the tract, take amp - base
                for pk in t:
                    pk -= 1  # peaks are 1-indexed
                    out.append(amplitude[pk] - _base)
        else:
            for amp, idx in zip(amplitude, index):
                # Get edge points of peak
                x = [idx[0], idx[-1]]
                y = [trace[x[0]], trace[x[-1]]]

                # Adjust heights if close to edge
                if abs(x[0] - 0) <= _edge_dist:
                    y[0] = y[-1]
                if abs(x[1] - len(trace) - 1) <= _edge_dist:
                    y[-1] = y[0]
                _base = np.mean(y)

                out.append(amp - _base)
        return out

    @staticmethod
    def _length(label: np.ndarray) -> List[int]:
        """"""
        peak, counts = np.unique(label[label > 0], return_counts=True)
        return list(counts)

    @staticmethod
    def _width(trace: np.ndarray,
               indexes: np.ndarray,
               tract: List[List[int]],
               amplitudes: List[float],
               prominences: List[float],
               relative: float,
               absolute: float = None
               ) -> List[float]:
        """"""
        # Peaks are the flat tracts
        peaks = [p for sl in tract for p in sl]

        # Get the target height
        if absolute:
            targets = [absolute] * len(peaks)
        else:
            targets = [a - (1 - relative) * p
                       for a, p in zip(amplitudes, prominences)]

        out = []
        for p, i, t, a in zip(peaks, indexes, targets, amplitudes):
            if t >= a:
                out.append(np.nan)
            else:
                # Calculate crossing points
                # crosses is pt before the crossing on both the way up
                # and the way down
                all_crosses = np.where(np.diff(np.sign(trace - t)))[0]
                peak_crosses = np.array([a for a in all_crosses if a in i])

                # Need to interpolate to find crosses between points
                direction = np.array([np.sign(t - trace[c])
                                      for c in peak_crosses])
                true_crosses = [c + ((t - trace[c]) / (trace[c + 1] - trace[c]))
                                for c in peak_crosses]
                ups = np.where(direction >= 0)[0]
                downs = np.where(direction < 0)[0]

                if len(ups) and len(downs):
                    # Found both
                    # Return first up cross and last down cross
                    out.append(true_crosses[downs[-1]] - true_crosses[ups[0]])
                else:
                    '''
                    This is where I would copy over the implementation of linear
                    estimation of crossing points that I already wrote. BUT, it
                    relies on having the slopes defined, which isn't hard to do
                    but I'm lazy and not doing it now. So for now, all that will
                    come out is nan if part of the peak is missing
                    '''
                    out.append(np.nan)

        return out

    @staticmethod
    def _area_under_curve(trace: np.ndarray,
                          label: np.ndarray
                          ) -> List[float]:
        """"""
        out = []
        for l in np.unique(label[label > 0]):
            out.append(integrate.simps(trace[label == l]))

        return out

    @staticmethod
    def _linearity(trace: np.ndarray,
                   label: np.ndarray
                   ) -> List[float]:
        """"""
        out = []
        for l in np.unique(label[label > 0]):
            idx = label == l
            x = np.arange(idx.sum())
            y = trace[idx]

            r, p = stats.pearsonr(x, y)
            out.append(1 / np.abs(r))
        return out
