import warnings
from typing import Tuple

import numpy as np

from celltk.utils._types import Track, Mask, Image, Arr
from celltk.core.operation import BaseEvaluator
from celltk.utils.utils import ImageHelper
from celltk.utils.operation_utils import track_to_mask
from celltk.utils.info_utils import nan_helper_1d


class Evaluator(BaseEvaluator):
    @ImageHelper(by_frame=False)
    def save_kept_cells(self,
                        track: Track,
                        array: Arr
                        ) -> Track:
        """"""
        # Figure out all the cells that were kept
        kept_cells = np.unique(array[:, :, 'label']).astype(int)

        # Change to mask to also blank negatives
        ravel = track_to_mask(track).ravel()

        # Remove the missing cells by excluding from mapping
        mapping = {c: c for c in kept_cells}
        ravel = np.asarray([mapping.get(c, 0) for c in ravel])

        # Add back the parent labels
        parent_ravel = track.ravel()  # Includes negative values
        mask = (parent_ravel < 0) * (ravel > 0)
        np.copyto(ravel, parent_ravel, where=mask)

        return ravel.reshape(track.shape)

    @ImageHelper(by_frame=False, as_tuple=False)
    def make_single_cell_stack(self,
                               image: Image,
                               array: Arr,
                               cell_id: int,
                               position_id: int = None,
                               window_size: Tuple[int] = (40, 40),
                               region: str = None,
                               channel: str = None
                               ) -> Image:
        """
        Should make a montage and follow the centroid of a single cell

        NOTE:
            - Cells very close to edge might raise IndexError
        """
        # Simpler if it's limited to even numbers only
        assert all([not (w % 2) for w in window_size])

        # Find the centroid for the cell in question
        region = array.regions[0] if not region else region
        channel = array.channels[0] if not channel else channel
        cell_index = np.where(array[region, channel, 'label'] == cell_id)[0]
        if len(np.unique(cell_index)) > 1:
            # Greater than one instance found
            if position_id:
                # Get it based on the position
                pass
            else:
                warnings.warn('Found more than one matching cell. Using '
                              f'first instance found at {cell_index[0]}.')
                cell_index = int(cell_index[0])
        else:
            cell_index = int(cell_index[0])

        # Get the centroid, window for the cell, and img size
        y, x = array[region, channel, ('y', 'x'), cell_index, :]
        y = nan_helper_1d(y)
        x = nan_helper_1d(x)
        frames, y_img, x_img = image.shape
        x_win, y_win = window_size

        # Make the window with the cell in the center of the window
        x_adj = int(x_win / 2)
        y_adj = int(y_win / 2)
        y_min = np.floor(np.clip(y - y_adj, a_min=0, a_max=None)).astype(int)
        y_max = np.floor(np.clip(y + y_adj, a_min=None, a_max=y_img)).astype(int)
        x_min = np.floor(np.clip(x - x_adj, a_min=0, a_max=None)).astype(int)
        x_max = np.floor(np.clip(x + x_adj, a_min=None, a_max=x_img)).astype(int)

        out = np.empty((frames, y_win, x_win), dtype=image.dtype)
        for fr in range(frames):
            fr = int(fr)
            out[fr, ...] = image[fr, y_min[fr]:y_max[fr], x_min[fr]:x_max[fr]]

        return out
