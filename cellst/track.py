import os
from typing import Tuple

import numpy as np
import btrack
import btrack.utils as butils
import btrack.constants as bconstants

from cellst.core.operation import BaseTracker
from cellst.utils._types import Image, Mask, Track
from cellst.utils.utils import ImageHelper, stdout_redirected
from cellst.utils.operation_utils import lineage_to_track

# Tracking algorithm specific imports
from kit_sch_ge.tracker.extract_data import get_indices_pandas
from cellst.utils.kit_sch_ge_utils import (TrackingConfig, MultiCellTracker,
                                           ExportResults)
from cellst.utils.bayes_utils import (bayes_extract_tracker_data,
                                      bayes_update_mask)


class Tracker(BaseTracker):
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
        # Add context management to supress printing to terminal
        # TODO: make this optional, log to file
        with stdout_redirected():
            tracks = tracker()

            exporter = ExportResults(postprocessing_key)
            mask, lineage = exporter(tracks, img_shape=img_shape,
                                     time_steps=list(range(image.shape[0])))

        return lineage_to_track(mask, lineage)

    @ImageHelper(by_frame=False)
    def simple_bayesian_track(self,
                              mask: Mask,
                              config_path: str = 'cellst/config/bayes_config.json',
                              update_method: str = 'exact',
                              ) -> Track:
        """
        Wraps BayesianTracker: https://github.com/quantumjot/BayesianTracker

        TODO:
            - Set values in config_file
            - Speed up bayes_extract_tracker_data
            - Add display with navari
            - Supress output with stdout_redirected()
        """
        # Convert mask to useable objects in btrack
        objects = butils.segmentation_to_objects(mask,
                                                 use_weighted_centroid=False)
        bayes_id_mask = bayes_update_mask(mask, objects)

        # Track as shown in btrack example
        with btrack.BayesianTracker() as tracker:
            tracker.configure_from_file(os.path.abspath(config_path))
            tracker.update_method = getattr(bconstants.BayesianUpdates,
                                            update_method.upper())
            tracker.append(objects)
            tracker.track()
            tracker.optimize()

            # Get the completed lineage
            # Order: label, frame0, frame1, parent, parent_track, tree depth
            lineage = np.vstack(tracker.lbep)[:, :4]

            # Convert mask labels before writing lineage
            new_mask = bayes_extract_tracker_data(bayes_id_mask, tracker)

        return lineage_to_track(new_mask, lineage)
