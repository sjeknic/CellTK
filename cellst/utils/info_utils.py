from typing import Collection

import numpy as np

SEED = 69420
_RNG = np.random.default_rng(SEED)


def get_bootstrap_population(arr: np.ndarray,
                             boot_reps: int = 1000
                             ) -> np.ndarray:
    """


    Args:
    Arr: response of cells in one condition, cells x response/times
    boot_reps: Number of bootstrap replicates

    Return:
    returns array boot_reps x response/times
    """
    boot_arrs = [_RNG.choice(arr, size=arr.shape[0], replace=True)
                 for _ in range(boot_reps)]
    arr = np.vstack([b.mean(0) for b in boot_arrs])

    return arr



# VVVVVVVVVVVVVVV Is copied directly from cell_info_theory


def mutual_info_from_contingency(self, contingency: np.ndarray) -> float:
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

def mutual_info_from_joint_prob(self, prob_arr: np.ndarray) -> float:

    h_x = self.shannon_entropy(prob_arr.sum(axis=1))
    h_y = self.shannon_entropy(prob_arr.sum(axis=0))
    h_xy = self.shannon_entropy(prob_arr)

    return h_x + h_y - h_xy

def shannon_entropy(self, vec: np.ndarray) -> float:
    vec_norm = vec / vec.sum()
    vec_norm = vec_norm[np.nonzero(vec_norm)]
    return -np.sum((vec_norm * np.log(vec_norm))) / np.log(2)

def blahut_arimoto(self,
                   arr: np.ndarray,
                   tol: float = 1e-7,
                   max_iter: int = 10000
                   ) -> Collection[np.ndarray]:
    """
    arr is contingency array (not joint probability)

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

    C = np.nansum(p_hat[:, np.newaxis] * arr * np.log(q_arr / p_hat[:, np.newaxis]))

    return C / np.log(2), p_hat

def randomize_array_by_row(self,
                           arr: np.ndarray
                           ) -> np.ndarray:

    rand_arr = np.zeros(arr.shape)

    # Randomize each row independently to keep same number of samples
    for n, a in enumerate(arr):
        rand_arr[n, :] = self._rng.multinomial(a.sum(), np.ones(len(a)) / len(a), size=1)

    return rand_arr

def randomize_array(self,
                    arr: np.ndarray
                    ) -> np.ndarray:

    samples = arr.sum()
    flat_len = len(arr.flatten())
    rand_samples = self._rng.multinomial(samples, np.ones(flat_len) / flat_len)

    return rand_samples.reshape(arr.shape)

def subset_array(self,
                 arr: np.ndarray,
                 subset: float = 0.6,
                 ) -> np.ndarray:
    '''
    Inputs:
        arr: contingency array of signal x response

    TODO:
        - Is there a way to do this without the loop?
    '''

    subset_arr = arr.copy()
    cells_to_remove = int(np.floor((1 - subset) * arr.sum()))

    while cells_to_remove > 0:

        # Probabilities for multinomial
        prob_arr = np.ones(subset_arr.shape)
        prob_arr[subset_arr == 0] = 0

        # Randomly assign cells to remove
        remove_arr = self._rng.multinomial(cells_to_remove, prob_arr.flatten() / prob_arr.sum(), size=1)
        remove_arr = remove_arr.reshape(subset_arr.shape)

        # Remove cells.
        subset_arr = subset_arr - remove_arr

        # If any went negative, set to 0 and rerun
        cells_to_remove = np.abs(subset_arr[subset_arr < 0].sum())
        subset_arr[subset_arr < 0] = 0

    return subset_arr