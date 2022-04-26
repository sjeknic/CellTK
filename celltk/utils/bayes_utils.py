import numpy as np

from celltk.utils._types import Mask


def bayes_update_mask(mask: Mask, objects: list) -> np.ndarray:
    """
    Change labels in mask to match ID given in objects

    btrack 0-indexes labels, so background value has to be changed
    So background will be -1

    Assume that regionprops goes through labels in order
    """
    out = np.ones(mask.shape, dtype=np.int32) * -1
    # Iterate through all frames in mask
    for t, frame in enumerate(mask):
        old_labels = np.unique(frame[frame > 0])
        obs = [o.ID for o in objects
               if o.t == t and not o.dummy]

        assert len(old_labels) == len(obs)

        for old, new in zip(old_labels, obs):
            out[t, ...][frame == old] = new

    return out


def bayes_extract_tracker_data(mask: Mask,
                               tracks: np.ndarray,
                               references: np.ndarray
                               ) -> Mask:
    """Updates labels in the given mask to match tracker references.
    bayesian_tracker first uniquely identifies all objects (mask). Then
    assigns real labels (tracks) to match the unique labels (references).
    This function produces a new mask with tracks as the labels.
    """
    # Flatten the input array for easier search
    ravel = mask.ravel()

    # Build mapping dictionar
    mapping = {}
    for btrk, bref in zip(tracks, references):
        for b in bref:
            if b > 0: mapping[b] = btrk

    # Build output array and then reshape
    out = [mapping.get(r, 0) for r in ravel]
    return np.asarray(out).reshape(mask.shape)
