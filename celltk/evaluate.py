import warnings
from typing import Tuple

import numpy as np
import skimage.segmentation as segm

from celltk.utils._types import Mask, Image, Array
from celltk.core.operation import BaseEvaluate
from celltk.utils.utils import ImageHelper
from celltk.utils.operation_utils import track_to_mask, get_cell_index, nan_helper_1d


class Evaluate(BaseEvaluate):
    @ImageHelper(by_frame=False)
    def save_kept_cells(self,
                        track: Mask,
                        array: Array
                        ) -> Mask:
        """Creates a track from the input track
        that only includes the cells that are present in array."""
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

        return ravel.reshape(track.shape).astype(np.int16)

    @ImageHelper(by_frame=False, as_tuple=False)
    def make_single_cell_stack(self,
                               image: Image,
                               array: Array,
                               cell_id: int,
                               position_id: int = None,
                               window_size: Tuple[int] = (40, 40),
                               region: str = None,
                               channel: str = None
                               ) -> Image:
        """
        Crops a window around the coordinates of a single cell
        in array.
        """
        # Simpler if it's limited to even numbers only
        assert all([not (w % 2) for w in window_size])

        # Find the row that contains the cell data
        region = array.regions[0] if not region else region
        channel = array.channels[0] if not channel else channel
        label_array = array[region, channel, 'label']
        if position_id is not None:
            position_array = array[region, channel, 'position_id']
        else:
            position_array = None
        cell_index = get_cell_index(cell_id, label_array,
                                    position_id, position_array)

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

        # Crop the orig array and save in out array - shape must always match
        out = np.empty((frames, y_win, x_win), dtype=image.dtype)
        for fr in range(frames):
            fr = int(fr)
            out[fr, ...] = image[fr, y_min[fr]:y_max[fr], x_min[fr]:x_max[fr]]

        return out

    @ImageHelper(by_frame=True)
    def overlay_tracks(self,
                       image: Image,
                       track: Mask,
                       boundaries: bool = False,
                       mode: str = 'inner'
                       ) -> Image:
        """Overlays the labels of objects over the reference image."""
        if (track < 0).any():
            track = track_to_mask(track)
        if boundaries:
            track = segm.find_boundaries(track, mode=mode)
        return np.where(track > 0, track, image).astype(np.uint16)
