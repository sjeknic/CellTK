import numpy as np
import btrack

from celltk.utils._types import Mask


def bayes_update_mask(mask: Mask, objects: list) -> np.ndarray:
    """
    Change labels in mask to match ID given in objects

    btrack 0-indexes labels, so background value has to be changed
    So background will be -1

    Assume that regionprops goes through labels in order
    """
    out = np.ones(mask.shape, dtype=np.int16) * -1
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
                               tracker: btrack.BayesianTracker
                               ) -> Mask:
    """
    Update labels in mask to match the tracker references

    TODO:
        - This function seems surprisingly slow
    """
    out = np.zeros_like(mask)
    for btrk, bref in zip(tracker.tracks, tracker.refs):
        trk_obs = np.isin(mask, [b for b in bref if b >= 0])
        out[trk_obs] = btrk['ID']

    return out