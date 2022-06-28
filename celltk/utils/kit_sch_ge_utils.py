#! /usr/bin/env python
from typing import Tuple

import numpy as np
import pandas as pd

from celltk.utils._types import Image, Mask
from celltk.external.kit_sch_ge.tracker.tracking import (TrackingConfig, MultiCellTracker,
                                         Tracklet)
from celltk.external.kit_sch_ge.tracker.export import (ExportResults, catch_tra_issues,
                                       create_tracking_mask_image)
from celltk.external.kit_sch_ge.tracker.extract_data import get_indices_pandas
from celltk.external.kit_sch_ge.tracker.postprocessing import (add_dummy_masks, untangle_tracks,
                                               no_fn_correction, no_untangling)

"""
This file inherits some of the classes from kit_sch_ge to make
them work in this framework. In general, trying to keep all of
the variable names the same as in kit_sch_ge, even if that ends
up confusing in the context of CellST.
"""


class TrackingConfig(TrackingConfig):
    def __init__(self,
                 image: Image,
                 mask: Mask,
                 roi_box_size: int,
                 delta_t: int = 2,
                 cut_off_distance: Tuple = None,
                 allow_cell_division: bool = True,
                 **kwargs
                 ) -> None:
        # Basically re-written from TrackingConfig
        # Attempting to keep variable and function names unchanged
        self.img_files = image
        self.segm_files = mask

        self.time_steps = list(range(image.shape[0]))
        self.roi_box_size = roi_box_size
        self.delta_t = delta_t
        self.allow_cell_division = allow_cell_division
        if cut_off_distance is None:
            self.cut_off_distance = max(roi_box_size) / 2
        else:
            self.cut_off_distance = cut_off_distance

    def get_image_file(self, time_step: int) -> np.ndarray:
        return self.img_files[time_step, ...]

    def get_segmentation_masks(self, time_step: int) -> np.ndarray:
        return (self.segm_files[time_step, ...],
                get_indices_pandas(self.segm_files[time_step, ...]))


class MultiCellTracker(MultiCellTracker):
    def propagate_tracklets(self, time: int):
        image = self.config.get_image_file(time)
        """Everything from here down is directly copied without changes"""
        if self.img_shape is None:
            self.img_shape = image.shape
        segmentation, mask_indices = self.config.get_segmentation_masks(time)

        # initialize tracklets
        for m_id, mask in mask_indices.items():
            m = np.array(mask)
            box_shape = np.max(m, axis=-1) - np.min(m, axis=-1) + 1
            box_shape = [max(a, b) for a, b in zip(box_shape, self.config.roi_box_size)]
            self.tracklets[(time, m_id)] = Tracklet(m_id, time, mask, box_shape, self.delta_t)
            self.tracklets[(time, m_id)].init_img_patch(time, image)

        # update tracklets from previous time steps
        selected_tracklets = filter(lambda x: x[0] < time, self.tracklets.keys())
        for tracklet_key in selected_tracklets:
            tracklet = self.tracklets[tracklet_key]
            if (time - tracklet.t_start) <= self.delta_t:

                if time not in tracklet.time_steps:
                    tracklet.propagate(time, image)
                # add potential matching candidates
                coords = tracklet.roi.last_roi_crop_box(self.img_shape)
                mask_ids = segmentation[coords]
                mask_ids = np.unique(mask_ids[mask_ids > 0])
                if len(tracklet.roi.roi) > 1:
                    prev_roi = tracklet.roi.roi[sorted(tracklet.roi.roi.keys())[-2]].crop_box(self.img_shape)
                    mask_ids_prev_roi = segmentation[prev_roi]
                    mask_ids_prev_roi = np.unique(mask_ids_prev_roi[mask_ids_prev_roi > 0])
                    mask_ids = np.unique(np.hstack([mask_ids, mask_ids_prev_roi]))

                if len(mask_ids) > 0:
                    n_matching_candidates = {(time, n_id) for n_id in mask_ids}
                    tracklet.add_matching_candidates(n_matching_candidates)


class ExportResults(ExportResults):

    def __call__(self, tracks, img_shape, time_steps):
        self.time_steps = time_steps

        if self.postprocessing_key == 'nd':
            tracks = no_untangling(tracks)
            tracks = add_dummy_masks(tracks, img_shape)
        elif self.postprocessing_key == 'ns+l':
            tracks = untangle_tracks(tracks)
            tracks = no_fn_correction(tracks, keep_link=True)
        elif self.postprocessing_key == 'ns-l':
            tracks = untangle_tracks(tracks)
            tracks = no_fn_correction(tracks, keep_link=False)
        elif self.postprocessing_key == 'nd_ns+l':
            tracks = no_untangling(tracks)
            tracks = no_fn_correction(tracks, keep_link=True)
        elif self.postprocessing_key == 'nd_ns-l':
            tracks = no_untangling(tracks)
            tracks = no_fn_correction(tracks, keep_link=False)
        else:
            tracks = untangle_tracks(tracks)
            tracks = add_dummy_masks(tracks, img_shape)

        tracks = catch_tra_issues(tracks, time_steps)

        return (self.create_segm_masks(tracks, img_shape),
                self.create_lineage_file(tracks))

    def create_lineage_file(self, tracks):
        """
        Returns lineage information as a numpy array?? pandas df??
        """
        track_info = {'track_id': [], 't_start': [], 't_end': [], 'predecessor_id': []}
        for t_id in sorted(tracks.keys()):
            track_data = tracks[t_id]
            track_info['track_id'].append(track_data.track_id)
            frame_ids = sorted(list(track_data.masks.keys()))
            track_info['t_start'].append(frame_ids[0])
            track_info['t_end'].append(frame_ids[-1])

            if isinstance(track_data.pred_track_id, list):
                if len(track_data.pred_track_id) > 0:
                    track_data.pred_track_id = track_data.pred_track_id[0]
                else:
                    track_data.pred_track_id = 0  # no predecessor

            track_info['predecessor_id'].append(track_data.pred_track_id)
        df = pd.DataFrame.from_dict(track_info)

        return df.to_numpy()

    def create_segm_masks(self, all_tracks, img_shape):
        tracks_in_frame = {}

        # create for each time step dict entry, otherwise missing time steps possible -> no img exported
        for t_step in self.time_steps:
            if t_step not in tracks_in_frame:
                tracks_in_frame[t_step] = []

        for track_data in all_tracks.values():
            time_steps = sorted(list(track_data.masks.keys()))
            for t_step in time_steps:
                if t_step not in tracks_in_frame:
                    tracks_in_frame[t_step] = []
                tracks_in_frame[t_step].append(track_data.track_id)

        t_max = sorted(list(tracks_in_frame.keys()))[-1]
        all_masks = np.empty((len(self.time_steps), *img_shape))

        # NOTE: idx is different than time. time can be non-consecutive
        for idx, (time, track_ids) in enumerate(tracks_in_frame.items()):
            tracking_mask = create_tracking_mask_image(all_tracks, time, track_ids, img_shape)
            tracking_mask = np.array(np.squeeze(tracking_mask), dtype=np.uint16)
            all_masks[idx, ...] = tracking_mask

        return all_masks
