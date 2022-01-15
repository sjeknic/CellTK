from typing import Collection, Tuple

import numpy as np
import btrack

from cellst.operation import BaseTrack
from cellst.utils._types import Image, Mask, Track
from cellst.utils.utils import ImageHelper, stdout_redirected

# Needed for Track.kit_sch_ge_track
from kit_sch_ge.tracker.extract_data import get_indices_pandas
from cellst.utils.kit_sch_ge_utils import (TrackingConfig, MultiCellTracker,
                                           ExportResults)
from cellst.utils.operation_utils import lineage_to_track, bayes_track_to_mask


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
        # Add context management to supress printing to terminal
        # TODO: make this optional
        with stdout_redirected():
            tracks = tracker()

            exporter = ExportResults(postprocessing_key)
            mask, lineage = exporter(tracks, img_shape=img_shape,
                                     time_steps=list(range(image.shape[0])))

        return lineage_to_track(mask, lineage)


    @ImageHelper(by_frame=False)
    def simple_bayesian_track(self,
                              mask: Mask,
                              config_path: str = 'bayes_config.json',
                              update_method: str = 'exact',
                              ) -> Track:
        """
        update_method can be EXACT or APPROXIMATE

        Wrapper for btrack.

        QUESTIONS:
            - Why is max_lost in the config file? Seems an arbitrary limit,
              that would also depend on the experiment/data.
            - Same for prob_not_assign, apopotosis_rate, segmentation_miss_rate, etc.
            - Can properties given to segmentation_to_objects() be used in
              the model. i.e. can I give them a state and incorporate them
              in the matrices somehow? How would they get used?
            - Do I always have to provide 3 coordinates? Can I give more???
        """
        # Convert mask to useable objects in btrack
        # TODO: Does adding extra measurements help here?
        objects = btrack.utils.segmentation_to_objects(mask)

        with btrack.BayesianTracker() as tracker:
            # TODO: Put config.json in a reasonable place
            tracker.configure_from_file(config_path)

            # TOOD: This should be an input option
            if update_method == 'approximate':
                tracker.update_method = btrack.constants.BayesianUpdates.APPROXIMATE
            else:
                tracker.update_method = btrack.constants.BayesianUpdates.EXACT

            tracker.append(objects)
            tracker.track_interactive()
            tracker.optimize()

            # Get the completed tracks
            # Order: label, frame0, frame1, parent, parent_track, tree depth
            lineage = np.vstack(tracker.lbep)[:, :4]

            # data is an array of label, frame, y, x
            data, properties, graph = tracker.to_napari(ndim=2)
            mask = bayes_track_to_mask(mask, data)

        return lineage_to_track(mask, lineage)


