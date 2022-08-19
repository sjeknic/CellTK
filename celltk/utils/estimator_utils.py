from typing import Tuple, Callable

import numpy as np
import scipy.stats as stats


"""None of these estimators need to be wrapped in partial or anything"""
def confidence_interval(arr: np.ndarray,
                        ci: float = 0.95,
                        ) -> np.ndarray:
    """Calculates the confidence interval based on a t-distribution.

    NOTE:
        - Only works on axis 0 for now
    """
    assert ci <= 1 and ci >= 0

    out = np.zeros([2, arr.shape[1]], dtype=float)
    dof = arr.shape[1] - 1
    # Calculate at each time point - so iterate columns
    for a in range(arr.shape[1]):
        col = arr[:, a]
        out[:, a] = stats.t.interval(ci, dof,
                                     loc=np.nanmean(col),
                                     scale=stats.sem(col, nan_policy='omit'))
    return out


def get_bootstrap_population(arr: np.ndarray,
                             boot_reps: int = 1000,
                             seed: int = 69420,
                             function: Callable = np.nanmean
                             ) -> np.ndarray:
    """

    Args:
        arr: response of cells in one condition, cells x response/times
        boot_reps: Number of bootstrap replicates

    Return:
        array boot_reps x response/times
    """
    _rng = np.random.default_rng(seed)
    boot_arrs = [_rng.choice(arr, size=arr.shape[0], replace=True)
                 for _ in range(boot_reps)]
    arr = np.vstack([function(b, axis=0) for b in boot_arrs])

    return arr


def bootstrap_estimator(arr: np.ndarray,
                        reps: int = 1000,
                        ci: float = 0.95,
                        axis: int = 0,
                        ignore_nans: bool = True,
                        function: Callable = np.nanmean
                        ) -> Tuple[np.ndarray]:
    """Uses bootstrap resampling to estimate a confidence interval.
    """
    assert ci <= 1 and ci >= 0
    ci *= 100

    # Sample the boostrap population
    boot = get_bootstrap_population(arr, reps, function=function)

    if ignore_nans:
        func = np.nanpercentile
    else:
        func = np.percentile

    # Calculate bounds
    low_end = func(boot, 50. - ci / 2., axis=axis, keepdims=True)
    hi_end = func(boot, 50. + ci / 2., axis=axis, keepdims=True)

    return np.vstack((hi_end, low_end))


def fraction_of_total(arr: np.ndarray,
                      ignore_nans: bool = True,
                      axis: int = 0
                      ) -> np.ndarray:
    """Returns the fraction of entries that have non-zero values.
    """
    if ignore_nans:
        return np.count_nonzero(arr[~np.isnan(arr)],
                                axis=axis, keepdims=True) / arr.shape[axis]
    else:
        return np.count_nonzero(arr, axis=axis, keepdims=True) / arr.shape[axis]


def wilson_score(arr: np.ndarray,
                 ci: float = 0.95,
                 axis: int = 0
                 ) -> np.ndarray:
    """Calculates the Wilson score for a binomial distribution.
    """
    # Do two-tailed by default, so divide by 2
    z = stats.norm.ppf(1 - (1 - ci) / 2)

    # Get proportion of population that is positive
    prob = np.squeeze(fraction_of_total(arr, axis=axis))

    # Get cell counts for each column
    n = (~np.isnan(arr)).sum(axis)

    # Calculate the wilson score
    denom = 1 + z ** 2 / n
    adj_prob = prob + z ** 2 / (2 * n)
    adj_sd = np.sqrt((prob * (1 - prob) + z ** 2 / (4 * n)) / n)

    hi = (adj_prob + z * adj_sd) / denom
    lo = (adj_prob - z * adj_sd) / denom

    return np.vstack([hi, lo])


def normal_approx(arr: np.ndarray,
                  ci: float = 0.95,
                  axis: int = 0
                  ) -> np.ndarray:
    """Calculates the normal error for a binomial distribution."""
    # Do two-tailed by default, so divide by 2
    z = stats.norm.ppf(1 - (1 - ci) / 2)

    # Get proportion of population that is positive
    prob = np.squeeze(fraction_of_total(arr, axis=axis))
    # Get cell counts
    n = (~np.isnan(arr)).sum(axis)

    return z * np.sqrt((prob * (1 - prob)) / n)
