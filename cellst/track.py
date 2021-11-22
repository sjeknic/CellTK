from typing import Collection, Tuple

import numpy as np

from cellst.operation import BaseTrack
from cellst.utils._types import Image, Mask, Track
from cellst.utils.utils import ImageHelper

# Needed for Track.kit_sch_ge_track
from kit_sch_ge.tracker.extract_data import get_indices_pandas
from cellst.utils.kit_sch_ge_utils import (TrackingConfig, MultiCellTracker,
                                           ExportResults)
from cellst.utils.operation_utils import lineage_to_track


class Track(BaseTrack):
    @ImageHelper(by_frame=False)
    def kit_sch_ge_track(self,
                         image: Image,
                         mask: Mask,
                         default_roi_size: int = 2,
                         delta_t: int = 2,
                         cut_off_distance: Tuple = None,
                         allow_cell_division: bool = True,
                         postprocessing_key: str = None,
                         ) -> Track:
        """
        See kit_sch_ge/run_tracking.py for reference

        TODO:
            - Use non-consecutive timesteps (mainly for naming of files)
            - Add saving of lineage file (probably in a separate run_operation function)
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
        track = lineage_to_track(mask, lineage)

        return track
