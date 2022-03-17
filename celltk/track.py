import os
import warnings
from typing import Tuple

import numpy as np
import btrack
import btrack.utils as butils
import btrack.constants as bconstants
import skimage.measure as meas
import skimage.segmentation as segm

from celltk.core.operation import BaseTracker
from celltk.utils._types import Image, Mask, Track
from celltk.utils.utils import ImageHelper, stdout_redirected
from celltk.utils.operation_utils import (lineage_to_track,
                                          match_labels_linear,
                                          voronoi_boundaries,
                                          get_binary_footprint)

# Tracking algorithm specific imports
from kit_sch_ge.tracker.extract_data import get_indices_pandas
from celltk.utils.kit_sch_ge_utils import (TrackingConfig, MultiCellTracker,
                                           ExportResults)
from celltk.utils.bayes_utils import (bayes_extract_tracker_data,
                                      bayes_update_mask)


class Tracker(BaseTracker):
    @ImageHelper(by_frame=False)
    def simple_linear_tracker(self,
                              mask: Mask,
                              voronoi_split: bool = True
                              ) -> Track:
        """
        Tracker based on frame-to-frame linear assignment

        voronoi_split keeps objects from merging, but slows
        down computation

        TODO: Multiple ways to improve this function
            - Add custom cost function
            - Add ability to use intensity information
        """
        # Iterate over all frames in mask
        for idx, fr in enumerate(mask):
            if not idx:
                # Make output arr and save first frame
                out = np.zeros_like(mask)
                out[idx, ...] = mask[idx]
            else:
                if voronoi_split:
                    # Mask borders on the frame before doing LAP
                    borders = voronoi_boundaries(out[idx - 1, ...], thick=True)
                    fr = fr.copy()
                    fr[borders] = 0
                    fr = meas.label(fr)

                out[idx, ...] = match_labels_linear(out[idx - 1, ...], fr)

        return out

    @ImageHelper(by_frame=False)
    def simple_watershed_tracker(self,
                                 mask: Mask,
                                 connectivity: int = 2,
                                 watershed_line: bool = True,
                                 keep_seeds: bool = False,
                                 ) -> Track:
        """
        Use segm.watershed serially

        keep_seeds: if True, seeds from previous frame will be
        kept in the current segmentation. This is more robust
        if the segmentation is incomplete, missing, or fragmented in some
        frames. However, it also means the mask will be monotonically
        increasing in size. It is not appropriate for images with a lot
        of drift or for moving objects.
        """
        # Iterate over all frames
        for idx, fr in enumerate(mask):
            if not idx:
                # Make output arr and save first frame
                out = np.zeros_like(mask)
                out[idx, ...] = mask[idx]
            else:
                # The last frame serves as the seeds
                seeds = out[idx - 1, ...]
                footprint = get_binary_footprint(connectivity)

                # Generate mask of relevant area
                if keep_seeds:
                    # Include pixels in this OR last image
                    mask = np.logical_or(fr, seeds)
                else:
                    # Only pixels in this frame can be tracked
                    mask = fr.astype(bool)

                # Fill watershed and save
                out[idx, ...] = segm.watershed(fr, seeds, footprint, mask=mask,
                                               watershed_line=watershed_line)

        return out

    @ImageHelper(by_frame=False)
    def kit_sch_ge_tracker(self,
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
        # If nothing is in mask, return an empty stack
        if not mask.sum():
            return np.zeros_like(mask)

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

        try:
            tracker = MultiCellTracker(config)
            # Add context management to supress printing to terminal
            # TODO: make this optional, log to file
            with stdout_redirected():
                tracks = tracker()

                exporter = ExportResults(postprocessing_key)
                mask, lineage = exporter(tracks, img_shape=img_shape,
                                         time_steps=list(range(image.shape[0])))
        except ValueError as e:
            warnings.warn(f'Tracking failed with ValueError {e}. Returning empty.')
            return np.zeros_like(mask)

        return lineage_to_track(mask, lineage)

    @ImageHelper(by_frame=False)
    def bayesian_tracker(self,
                         mask: Mask,
                         config_path: str = 'celltk/config/bayes_config.json',
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
