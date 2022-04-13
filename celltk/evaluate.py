import numpy as np

from celltk.utils._types import Track, Mask, Image, Arr
from celltk.core.operation import BaseEvaluator
from celltk.utils.utils import ImageHelper
from celltk.utils.operation_utils import track_to_mask


class Evaluator(BaseEvaluator):
    @ImageHelper(by_frame=False)
    def save_kept_cells(self,
                        track: Track,
                        array: Arr
                        ) -> Track:
        """"""
        # Figure out all the cells that were missing
        all_cells = np.unique(track)
        kept_cells = np.unique(array[:, :, 'label']).astype(int)
        missing_cells = set(all_cells).difference(kept_cells)

        # Change to mask to also blank negatives
        out = track_to_mask(track)
        for cell in missing_cells:
            out[track == cell] = 0

        # Add back the parent labels
        parents = np.unique(track[track < 0])
        for par in parents:
            np.where(np.logical_and(track == par, track > 0), par, out)

        return out
