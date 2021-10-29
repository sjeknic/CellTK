from typing import Collection, Tuple

import numpy as np

from cellst.operation import BaseTrack
from cellst.utils._types import Image, Mask, Track
from cellst.utils.utils import image_helper

# Needed for Track.kit_sch_ge_track
from kit_sch_ge.tracker.extract_data import get_indices_pandas
from cellst.utils.kit_sch_ge_utils import (TrackingConfig, MultiCellTracker,
                                           ExportResults)


class Track(BaseTrack):
    def track_to_lineage(self, track: Track, lineage: np.ndarray):
        """
        Given a set of track images, reconstruct all the lineages

        TODO:
            - This function might be more appropriate in Extract
        """
        pass

    def lineage_to_track(self,
                         mask: Mask,
                         lineage: np.ndarray
                         ) -> Track:
        """
        Each mask in each frame should have a random(?) pixel
        set to the negative value of the parent cell.

        TODO:
            - This is less reliable, and totally lost, with small regions
            - must check that it allows for negatives
        """
        out = mask.copy().astype(np.int16)
        for (lab, app, dis, par) in lineage:
            if par:
                # Get all pixels in the label
                lab_pxl = np.where(mask[app, ...] == lab)

                # Find the centroid and set to the parent value
                # TODO: this won't work in all cases. trivial example if size==1
                x = int(np.floor(np.sum(lab_pxl[0]) / len(lab_pxl[0])))
                y = int(np.floor(np.sum(lab_pxl[1]) / len(lab_pxl[1])))
                out[app, x, y] = -1 * par

        return out

    @image_helper
    def kit_sch_ge_track(self,
                         image: Image,
                         mask: Mask,
                         default_roi_size: int = 2,
                         delta_t: int = 2,
                         cut_off_distance: Tuple = None,
                         allow_cell_division: bool = True,
                         postprocessing_key: str = None
                         ) -> Track:
        """
        See kit_sch_ge/run_tracking.py for reference

        TODO:
            - Use non-consecutive timesteps (mainly for naming of files)
            - Add saving of lineage file (probably in a separate run_operation function)
            - Add create Tracks from lineage
        """

        assert image.shape == mask.shape, f'Image/Mask mismatch {image.shape} {mask.shape}'

        img_shape = mask[-1, ...].shape
        masks = get_indices_pandas(mask[-1, ...])
        m_shape = np.stack(masks.apply(
            lambda x: np.max(np.array(x), axis=-1) - np.min(np.array(x), axis=-1) + 1
            ))

        if len(img_shape) == 2:
            if len(masks) > 10:
                m_size = np.median(np.stack(m_shape)).astype(int)

                roi_size = tuple([m_size*default_roi_size, m_size*default_roi_size])
            else:
                roi_size = tuple((np.array(img_shape) // 10).astype(int))
        else:
            roi_size = tuple((np.median(np.stack(m_shape), axis=0) * default_roi_size).astype(int))

        config = TrackingConfig(image, mask, roi_size, delta_t=delta_t,
                                cut_off_distance=cut_off_distance,
                                allow_cell_division=allow_cell_division)

        tracker = MultiCellTracker(config)
        tracks = tracker()

        exporter = ExportResults(postprocessing_key)
        mask, lineage = exporter(tracks, img_shape=img_shape, time_steps=list(range(image.shape[0])))
        track = self.lineage_to_track(mask, lineage)

        return track
