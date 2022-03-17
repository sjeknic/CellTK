from typing import Tuple

import numpy as np
import scipy.stats as stats

from celltk.utils.info_utils import get_bootstrap_population

"""None of these estimators need to be wrapped in partial or anything"""
def bootstrap_estimator(arr: np.ndarray,
                        reps: int = 1000,
                        ci: float = 0.95,
                        ax: int = 0,
                        ignore_nans: bool = True
                        ) -> Tuple[np.ndarray]:
    """"""
    assert ci <= 1 and ci >= 0
    ci *= 100

    boot = get_bootstrap_population(arr, reps)

    if ignore_nans:
        func = np.nanpercentile
    else:
        func = np.percentile

    low_end = func(boot, 50. - ci / 2., axis=ax, keepdims=True)
    hi_end = func(boot, 50. + ci / 2., axis=ax, keepdims=True)

    return np.vstack((low_end, hi_end))


def fraction_of_total(arr: np.ndarray,
                      ignore_nans: bool = True,
                      ax: int = 0
                      ) -> np.ndarray:
    """"""
    if ignore_nans:
        return np.nansum(arr, axis=ax, keepdims=True) / arr.shape[0]
    else:
        return np.sum(arr, axis=ax, keepdims=True) / arr.shape[0]