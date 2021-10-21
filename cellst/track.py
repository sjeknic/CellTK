from typing import Collection, Tuple

import numpy as np

from cellst.operation import Operation
from cellst.utils._types import Image, Mask, Track
from cellst.utils.utils import image_helper

# Needed for Track.kit_sch_ge_track
from kit_sch_ge.tracker.extract_data import get_indices_pandas
from cellst.utils.kit_sch_ge_utils import (TrackingConfig, MultiCellTracker,
                                           ExportResults)


class Track(Operation):
    _input_type = (Image, Mask)
    _output_type = Track

    def __init__(self,
                 input_images: Collection[str] = [],
                 input_masks: Collection[str] = [],
                 output: str = 'track',
                 save: bool = False,
                 track_file: bool = True,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        if isinstance(input_masks, str):
            self.input_masks = [input_masks]
        else:
            self.input_masks = input_masks

        self.output = output

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
        lineage = exporter(tracks, img_shape=img_shape, time_steps=list(range(image.shape[0])))

