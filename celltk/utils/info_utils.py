import functools
import warnings
from typing import Collection, Tuple, Union, Callable, List

import numpy as np
import scipy.stats as stats
import sklearn.preprocessing as skpre
import sklearn.neighbors as neigh
import sklearn.decomposition as decomp
import sklearn.cluster as clust
import skimage.util as util
import hdbscan
import umap

import matplotlib.pyplot as plt
import matplotlib

from celltk.utils._fastkde import FastLaplacianKDE


SEED = 69420
_RNG = np.random.default_rng(SEED)


### Functions for calculating information metrics ###
def shannon_entropy(vec: np.ndarray) -> float:
    vec_norm = vec / vec.sum()
    vec_norm = vec_norm[np.nonzero(vec_norm)]
    return -np.sum((vec_norm * np.log(vec_norm))) / np.log(2)


def mutual_info_from_joint_prob(prob_arr: np.ndarray) -> float:

    h_x = shannon_entropy(prob_arr.sum(axis=1))
    h_y = shannon_entropy(prob_arr.sum(axis=0))
    h_xy = shannon_entropy(prob_arr)

    return h_x + h_y - h_xy


def mutual_info_from_contingency(contingency: np.ndarray) -> float:
    # Taken from sklearn.mutual_info_score
    # Using np.log instead of math.log and removing contingency checks

    # Check about using sklearn.mutual_info_score again

    contingency_sum = contingency.sum()
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum

    outer = (pi.take(nzx).astype(np.int64, copy=False)
             * pj.take(nzy).astype(np.int64, copy=False))

    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - np.log(contingency_sum)) +
          contingency_nm * log_outer)

    return np.clip(mi.sum(), 0.0, None) / np.log(2)


def blahut_arimoto(arr: np.ndarray,
                   tol: float = 1e-7,
                   max_iter: int = 10000
                   ) -> Collection[np.ndarray]:
    """Runs Blahut-Arimoto algorithm

    Blahut-Arimoto is an algorithm for optimizing the
    input marginal probability distribution to a joint probability array

    TODO:
        - Return mi_inf estimate instead of joint_prob estimate
            - requires some way to keep track of the contingency array...
    """

    # Assume uniform input marginal to initiate
    p_hat = np.ones(arr.shape[0])
    p_hat = p_hat / p_hat.sum()
    q_arr = np.zeros(arr.shape)

    # Check for negative entries - shouldn't ever happen
    if arr[arr < 0]:
        print('Negative entries in matrix.')
        return np.nan, np.nan

    # Remove columns with all 0
    if (np.sum(arr, axis=0) == 0).any():
        arr = arr[:, ~(arr == 0).T.all(1)]

    # Normalize array if needed
    if not (np.sum(arr, axis=1) == 1).all():
        arr = arr / arr.sum(axis=1)[:, np.newaxis]

    # Run iterative Blahut-Arimoto algorithm
    it = 0
    while it < max_iter:
        it += 1

        q_arr = p_hat[:, np.newaxis] * arr
        q_arr = q_arr / q_arr.sum(axis=0)[np.newaxis, :]

        p_new = np.prod(np.power(q_arr, arr), axis=1)
        p_new = p_new / p_new.sum()

        if np.sum(np.abs(p_hat - p_new)) < tol:
            break
        else:
            p_hat = p_new.copy()

    # Calculate channel capacity
    C = np.nansum(p_hat[:, np.newaxis] * arr * np.log(q_arr / p_hat[:, np.newaxis]))

    return C / np.log(2), p_hat


### Functions for modifying arrays ###
def make_contingency_array(arrays: Collection[np.ndarray],
                           cluster_method: str = 'kmeans',
                           preprocess: str = None,
                           preprocess_kw: dict = {},
                           return_labels: bool = False,
                           **kwargs
                           ) -> np.ndarray:
    """
    """
    # Get indexes defining each array
    split_idxs = get_split_idxs(arrays)
    raw_arr = np.vstack(arrays)

    # Reduce dimensions if needed
    if preprocess:
        raw_arr = preprocess_array(raw_arr, **preprocess_kw)

    # Cluster array and label individual cells
    # TODO: Messy, move to its own function
    if cluster_method == 'kmeans':
        kws = {
            'n_clusters': 5
        }
        kws.update(kwargs)

        clusterer = clust.KMeans(**kws).fit(raw_arr)
    elif cluster_method == 'agglom':
        kws = {
            'n_clusters': 10
        }
        kws.update(kwargs)

        clusterer = clust.AgglomerativeClustering(**kws).fit(raw_arr)
    elif cluster_method == 'hdbscan':
        kws = {
            'cluster_selection_method': 'eom',
            'min_cluster_size': 15,
            'min_samples': 1,
            'cluster_selection_epsilon': 0,
            'alpha': 1.  # Shouldn't need to change this
        }
        kws.update(kwargs)

        clusterer = hdbscan.HDBSCAN(**kws).fit(raw_arr)
    elif cluster_method == 'optics':
        kws = {
            'min_samples': 5,
            'metric': 'minkowski',
            'p': 2
        }
        kws.update(kwargs)
        clusterer = clust.OPTICS(**kws).fit(raw_arr)
    else:
        raise ValueError(f'Method {cluster_method} not understood.')

    _labels = np.array(clusterer.labels_)
    labels = split_array(_labels, split_idxs)

    # Make output array and sort cells - labels are 0-indexed
    contingency = np.zeros((len(arrays), np.max(_labels) + 1))
    for row, label in enumerate(labels):
        # Get the number of cells in each label
        # HDBSCAN can return negative labels for noise points
        idx, num = np.unique(label[label >= 0], return_counts=True)
        for idx, num in zip(idx, num):
            contingency[row, idx] = num

    if return_labels:
        return contingency, labels
    else:
        return contingency


def preprocess_array(array: Collection[np.ndarray],
                     method: str = 'pca',
                     **kwargs
                     ) -> Collection[np.ndarray]:
    """"""
    # Apply dimension reduction
    if method == 'pca':
        array = pca_reduction(array, **kwargs)
    elif method == 'umap':
        array = umap_reduction(array, **kwargs)

    return array


def pca_reduction(arr: np.ndarray,
                  n_components: int = 2,
                  copy: bool = True,
                  svd_solver: str = 'auto',
                  **kwargs
                  ) -> np.ndarray:

    fitter = decomp.PCA(n_components=n_components, copy=copy,
                        svd_solver=svd_solver, **kwargs)

    return fitter.fit_transform(arr)


def umap_reduction(arr: np.ndarray,
                   n_neighbors: int = 15,
                   min_dist: float = 0,
                   n_components: int = 2,
                   metric: str = 'euclidean',
                   low_memory: bool = False,
                   transform_seed: int = 42,
                   init: str = 'spectral',
                   **kwargs
                   ) -> np.ndarray:
    """
    Uses UMAP to project response array into lower dimensional space
    """
    fitter = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                       n_components=n_components, metric=metric,
                       low_memory=low_memory, verbose=False,
                       transform_seed=transform_seed, init=init,
                       **kwargs).fit(arr)
    return np.array(fitter.embedding_)


def normalize_arrays(arrays: Collection[np.ndarray],
                     method: str = 'indiv',
                     function: str = 'MinMaxScaler'
                     ) -> Collection[np.ndarray]:
    """
    Proposed normalization methods
    1) indiv - each feature is normalized independently in all the arrays
    2) combined - all features in all thes array are normalized to the same scale
    3) separate - all features are normalized independently in each array
    3.5) separate-combined - all features in each array is normalized to the same scale
    """
    # Check inputs
    n_feats = np.unique([s.shape[1] for s in arrays])
    if len(n_feats) == 1:
        n_feats = n_feats[0]
    else:
        raise ValueError(f'Got differing numbers of features. {n_feats}')

    # Get normalizing function from sklearn if no function given
    if isinstance(function, str):
        try:
            func = getattr(skpre, function)().fit_transform
        except AttributeError:
            raise AttributeError(f'Normalizing function {function} '
                                 'not found in sklearn.')

    # Save array sizes for stacking/unstacking
    split_idxs = get_split_idxs(arrays)

    # Apply normalization
    if method == 'indiv':
        # Stack all arrays and normalize together
        _temp = func(np.vstack(arrays))
        arrays = [n for n in np.split(_temp, split_idxs, axis=0)
                  if n.shape[0] > 0]
    elif method == 'combined':
        # Get all features into one column, then scale
        _temp = func(np.vstack(arrays).reshape(-1, 1)).reshape(-1, n_feats)
        arrays = [n for n in np.split(_temp, split_idxs, axis=0)
                  if n.shape[0] > 0]
    elif method == 'separate':
        # Apply normalization to each array individually
        arrays = [func(s) for s in arrays]
    elif method == 'separate-combined':
        # Get all features in each array in one col, then scale
        arrays = [func(s.reshape(-1, 1)).reshape(-1, n_feats)
                  for s in arrays]
    else:
        raise ValueError(f'Normalization method {method} not understood.')

    return arrays


def get_bootstrap_population(arr: np.ndarray,
                             boot_reps: int = 1000
                             ) -> np.ndarray:
    """

    Args:
        arr: response of cells in one condition, cells x response/times
        boot_reps: Number of bootstrap replicates

    Return:
        array boot_reps x response/times
    """
    boot_arrs = [_RNG.choice(arr, size=arr.shape[0], replace=True)
                 for _ in range(boot_reps)]
    arr = np.vstack([np.nanmean(b, 0) for b in boot_arrs])

    return arr


def subset_array(arr: np.ndarray,
                 subset: float = 0.6,
                 ) -> np.ndarray:
    """
    Inputs:
        arr: contingency array of signal x response
        subset: fraction of original cells to keep
    """
    subset_arr = arr.copy()
    cells_to_remove = int(np.floor((1 - subset) * arr.sum()))
    prob_arr = np.ones(subset_arr.shape)

    while cells_to_remove > 0:
        # Probabilities for multinomial
        prob_arr[subset_arr == 0] = 0

        # Randomly assign cells to remove
        remove_arr = _RNG.multinomial(cells_to_remove,
                                      prob_arr.flatten() / prob_arr.sum(),
                                      size=1)
        remove_arr = remove_arr.reshape(subset_arr.shape)

        # Remove cells.
        subset_arr = subset_arr - remove_arr

        # If any went negative, set to 0 and rerun
        cells_to_remove = np.abs(subset_arr[subset_arr < 0].sum())
        subset_arr[subset_arr < 0] = 0

    return subset_arr


def randomize_array(arr: Union[Collection[np.ndarray], np.ndarray]
                    ) -> np.ndarray:
    """"""
    # If multiple arrays, stack, randomize, unstack
    _splits = False
    if isinstance(arr, (list, tuple)):
        _splits = get_split_idxs(arr)
        arr = np.vstack(arr)

    # Randomly pull rows from the array
    samples = arr.sum()
    flat_len = len(arr.flatten())
    rand_samples = _RNG.multinomial(samples, np.ones(flat_len) / flat_len)
    rand_samples = rand_samples.reshape(arr.shape)

    if _splits:
        rand_samples = split_array(rand_samples, _splits)

    return rand_samples


def _nan_helper(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of nans in a 1D array."""
    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_helper_2d(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of nans along rows in 2D array.

    TODO:
        - Move to a more sensible util file
    """
    temp = np.zeros(arr.shape)
    temp[:] = np.nan
    for n, y in enumerate(arr.copy()):
        nans, z = _nan_helper(y)
        y[nans] = np.interp(z(nans), z(~nans), y[~nans])
        temp[n, :] = y

    return temp


def nan_helper_1d(arr: np.ndarray) -> np.ndarray:
    temp = arr.copy()
    nans, z = _nan_helper(arr)
    temp[nans] = np.interp(z(nans), z(~nans), arr[~nans])
    return temp


def get_split_idxs(arrays: Collection[np.ndarray], axis: int = 0) -> List[int]:
    """
    """
    row_idxs = [s.shape[axis] for s in arrays]
    split_idxs = [np.sum(row_idxs[:i + 1])
                  for i in range(len(row_idxs))]
    return split_idxs


def split_array(array: np.ndarray, split_idxs: List[int], axis: int = 0) -> List[np.ndarray]:
    """
    """
    return [n for n in np.split(array, split_idxs, axis=axis)
            if n.shape[axis] > 0]


### Functions for more complicated calculations ###
def estimate_mi_inf(contingency_array: np.ndarray,
                    subset_range: Collection[float] = [0.6, 1.0],
                    subset_steps: int = 5,
                    replicates: int = 20,
                    ) -> Tuple[(int, list)]:
    """
    Extrapolates calculated mutual information to the case
    where 1/N_samples is infinity.
    """
    subset_range = np.linspace(subset_range[0], subset_range[1], subset_steps)
    inverse_subset_Nt = 1 / (subset_range * contingency_array.sum())

    mi_set = []
    for n, sub in enumerate(subset_range):
        temp_set = []
        for _ in range(replicates):
            # Samples contingency array with replacement
            sub_array = subset_array(contingency_array)

            # TODO: Allow use of different marginal types
            temp_set.append(blahut_arimoto(sub_array)[0])
        mi_set.append(temp_set)

    # Intercept of best fit line is MI_inf
    mi_means = [np.mean(m) for m in mi_set]
    slope, intercept = np.polyfit(inverse_subset_Nt, mi_means, deg=1)

    return intercept, mi_set


def calculate_channel_capacity(contingency: np.ndarray,
                               return_marginal: bool = False,
                               ) -> float:
    """Calculates discrete channel capacity from contingency array"""
    capacity, p_hat = blahut_arimoto(contingency)

    if return_marginal:
        return capacity, p_hat
    else:
        return capacity


def calculate_differential_channel_capacity(arrays: Collection[np.ndarray],
                                            boot_reps: int = None,
                                            normalize: str = None,
                                            num_hashes: int = 1000,
                                            ) -> float:

    """Calculates differential mutual information using KDE estimator

    Args:
        num_hashes: Larger values lead to more accurate KDEs but longer comp time

    Returns:
        differential channel capacity

    This function should calculate the channel capacity using the differential entropies,
    instead of discrete entropies by clustering. The conditional and un-conditional differential
    entropies should be estimated from the given data using KDE.

    As was done in the paper, get the KDE for each trace as if it were from each other
    stimulus. Sum those KDEs, weighted by both the number of traces and number of stimuli.

    TODO: This function should be agnostic to the method of calculating the probability density
          estimate, that should be passed to this model separately.,
    TODO: Was this implemented correct? Gives very high answers.
    """
    # Warn about nans which will likely mess up calculations
    if any([np.isnan(a).sum() for a in arrays]):
        warnings.warn('Arrays contain nans.', UserWarning)

    # Adjust data inputs as needed
    if boot_reps:
        arrays = [get_bootstrap_population(arr) for arr in arrays]
    if normalize:
        arrays = normalize_arrays(arrays, normalize)

    # Estimate bandwidth from median distance to first neighbor
    all_knns = [neigh.NearestNeighbors(n_neighbors=1).fit(s) for s in arrays]
    bands = [np.median(knn.kneighbors(n_neighbors=1, return_distance=True)[0])
             for knn in all_knns]

    # Estimate KDEs for each array
    indiv_kdes = [FastLaplacianKDE(a, b, num_hashes)
                  for a, b in zip(arrays, bands)]
    num_cells = [s.shape[0] for s in arrays]

    # Assume equal signal probabilities
    q = 1 / len(arrays)
    total_cond = 0
    total_uncond = 0
    for s_idx in range(len(arrays)):
        # Get unconditional probabilities from ALL KDE estimators
        uncond_probs = [np.apply_along_axis(i.kde, 1, arrays[s_idx])
                        for i in indiv_kdes]

        # Conditional probability comes from KDE estimator for this array
        cond_probs = uncond_probs[s_idx]

        # Sum unconditional probabilities together
        uncond_probs = np.vstack([q * u for u in uncond_probs]).sum(0)

        # Assume each trace is equally likely for entropy calculation
        trace_prob = (1 / num_cells[s_idx])
        cond_entropy = -q * np.sum(np.log2(cond_probs) * trace_prob)
        uncond_entropy = -q * np.sum(np.log2(uncond_probs) * trace_prob)

        # Entropies sum together
        total_cond += cond_entropy
        total_uncond += uncond_entropy

    return total_uncond - total_cond
