"""Utilities to export tracking results to ctc format"""
import os

import numpy as np
import pandas as pd
from tifffile import imsave

from tracker.postprocessing import add_dummy_masks, untangle_tracks
from tracker.postprocessing import no_fn_correction, no_untangling


class ExportResults:
    def __init__(self, postprocessing_key=None):
        """
        Exports tracking results to ctc format.
        Args:
            postprocessing_key: optional string to remove post-processing steps,
             if none is provided both post-processing steps (untangling, FN correction) are applied.
              'nd': 'no untangling',
              'ns+l': 'no FN correction but keep link of fragmented track as predecessor-successor',
              'ns-l': 'no FN correction and no link',
              'nd_ns+l': 'no untangling  and no FN correction
                          but keep link of fragmented track as predecessor-successor',
              'nd_ns-l': 'no untangling and no FN correction and no link'
        """
        self.img_file_name = 'mask'
        self.img_file_ending = '.tif'
        self.track_file_name = 'res_track.txt'
        self.time_steps = None
        self.postprocessing_key = postprocessing_key

    def __call__(self, tracks, export_dir, img_shape, time_steps):
        """
        Post-processes a tracking result and exports it to the ctc format of tracking masks and a lineage file.
        Args:
            tracks: a dict containing the trajectories
            export_dir: a path where to store the exported tracking results
            img_shape: a tuple proving the shape for the tracking masks
            time_steps: a list of time steps
        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.time_steps = time_steps
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.time_steps = time_steps

        if self.postprocessing_key == 'nd':
            tracks = no_untangling(tracks)
            print('add dummy masks')
            tracks = add_dummy_masks(tracks, img_shape)
        elif self.postprocessing_key == 'ns+l':
            print('untangle')
            tracks = untangle_tracks(tracks)
            tracks = no_fn_correction(tracks, keep_link=True)
        elif self.postprocessing_key == 'ns-l':
            print('untangle')
            tracks = untangle_tracks(tracks)
            tracks = no_fn_correction(tracks, keep_link=False)
        elif self.postprocessing_key == 'nd_ns+l':
            tracks = no_untangling(tracks)
            tracks = no_fn_correction(tracks, keep_link=True)
        elif self.postprocessing_key == 'nd_ns-l':
            tracks = no_untangling(tracks)
            tracks = no_fn_correction(tracks, keep_link=False)
        else:
            print('untangle')
            tracks = untangle_tracks(tracks)
            print('add dummy masks')
            tracks = add_dummy_masks(tracks, img_shape)

        tracks = catch_tra_issues(tracks, time_steps)
        print('export masks')
        self.create_lineage_file(tracks, export_dir)
        self.create_segm_masks(tracks, export_dir, img_shape)

    def create_lineage_file(self, tracks, export_dir):
        """
        Creates the lineage file.
        Args:
            tracks:  a dict containing the trajectories
            export_dir: path to the folder where the results shall be stored
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
        df.to_csv(os.path.join(export_dir, self.track_file_name),
                  columns=["track_id", "t_start", "t_end", 'predecessor_id'],
                  sep=' ', index=False, header=False)

    def create_segm_masks(self, all_tracks, export_dir, img_shape):
        """
        Creates for each time step a tracking image with masks
        corresponding to the segmented and tracked objects.
        Args:
            all_tracks: a dict containing the trajectories
            export_dir: a path where to store the exported tracking results
            img_shape: a tuple proving the shape for the tracking masks
        """
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
        z_fill = np.int(np.ceil(max(np.log10(max(1, t_max)), 3)))  # either 3 or 4 digits long frame id
        for time, track_ids in tracks_in_frame.items():
            tracking_mask = create_tracking_mask_image(all_tracks, time, track_ids, img_shape)

            file_name = self.img_file_name + str(time).zfill(z_fill) + self.img_file_ending
            tracking_mask = np.array(np.squeeze(tracking_mask), dtype=np.uint16)
            imsave(os.path.join(export_dir, file_name), tracking_mask, compress=1)


def create_tracking_mask_image(all_tracks, time, track_ids, img_shape):
    """
    Constructs image containing tracking masks and resolves overlapping masks.
    Args:
        all_tracks: a dict containing the trajectories
        time: int indicating the time point
        track_ids: list of track ids at the selected time point
        img_shape: a tuple providing the image shape of the mask image

    Returns: an np.array with the tracking masks for a time point

    """
    all_masks = {}
    all_mask_center = []
    all_mask_ids = []
    tracking_mask = np.zeros((1, *img_shape), dtype=np.uint16)
    for t_id in track_ids:
        track = all_tracks[t_id]
        mask = track.masks[time]
        all_masks[t_id] = mask
        mask_median = np.median(mask, axis=-1)
        if not all_mask_center:
            all_mask_center.append(mask_median)
        elif (not np.any(np.all((mask_median == all_mask_center), axis=-1))) or (len(all_mask_center) == 0):
            all_mask_center.append(mask_median)
        else:
            dist = np.linalg.norm(np.array(mask) - mask_median.reshape(-1, 1), axis=0)
            sorted_ids = np.argsort(dist, axis=0)
            sorted_mask = np.array(mask)[:, sorted_ids]
            index_nearest_point = np.argmin([np.any(np.all(el == all_mask_center, axis=-1))
                                             for el in sorted_mask.transpose()])
            all_mask_center.append(sorted_mask[:, index_nearest_point])

        all_mask_ids.append(t_id)

        # due to interpolated masks: overlapping with other masks possible -> reassign overlapping pixels
        colliding_pixels = np.array([np.any(img_plane[mask] > 0, axis=0) for img_plane in tracking_mask])
        if np.all(colliding_pixels > 0):
            # add new plane
            tracking_mask = np.vstack([tracking_mask, np.zeros((1, *img_shape), dtype=np.uint16)])
            img_plane = tracking_mask[-1]
        else:
            # add colliding pixels to first plane without collision
            # split selection of plane and mask indices as otherwise mask indices considered matrix
            # as p reference on tracking mask, tracking mask is edited
            img_plane = tracking_mask[np.argmax(colliding_pixels == 0)]
        img_plane[mask] = t_id
    if tracking_mask.shape[0] > 1:  # colliding pixels
        is_collision = np.sum(tracking_mask > 0, axis=0) > 1
        single_plane = tracking_mask.copy()
        single_plane[:, is_collision] = 0
        single_plane = np.sum(single_plane, axis=0)
        all_mask_ids = np.array(all_mask_ids)
        all_mask_center = np.array(all_mask_center)
        ind_pixel = list(zip(*np.where(is_collision)))
        pixel_masks = tracking_mask[:, is_collision].T
        for pixel_ind, masks_pixel in zip(ind_pixel, pixel_masks):
            # sort as all_m_ids sorted as well-> otherwise swaps in m_ids possible
            m_ids = sorted(masks_pixel[masks_pixel > 0])
            mask_centers = all_mask_center[np.isin(all_mask_ids, m_ids)]
            dist = np.sqrt(np.sum(np.square(mask_centers - np.array(pixel_ind).reshape(1, -1)), axis=-1))
            single_plane[pixel_ind] = m_ids[np.argmin(dist)]

        # add unmerged masks to tracking masks - for each tracked object a segm masks now in img
        tracking_mask = single_plane
    return tracking_mask


def catch_tra_issues(tracks, time_steps):
    """
    Adds for each empty tracking frame the tracking result of the temporally closest frame.
    Otherwise CTC measure can yield an error.
    Args:
        tracks: a dict containing the tracking results
        time_steps: a list of time steps

    Returns: the modified tracks

    """
    tracks_in_frame = {}
    for track_data in tracks.values():
        track_timesteps = sorted(list(track_data.masks.keys()))
        for t_step in track_timesteps:
            if t_step not in tracks_in_frame:
                tracks_in_frame[t_step] = []
            tracks_in_frame[t_step].append(track_data.track_id)
    if sorted(time_steps) != sorted(list(tracks_in_frame.keys())):
        empty_timesteps = sorted(np.array(time_steps)[~np.isin(time_steps, list(tracks_in_frame.keys()))])
        filled_timesteps = np.array(sorted(list(tracks_in_frame.keys())))
        for empty_frame in empty_timesteps:
            nearest_filled_frame = filled_timesteps[np.argmin(abs(filled_timesteps-empty_frame))]
            track_ids = tracks_in_frame[nearest_filled_frame]
            for track_id in track_ids:
                tracks[track_id].masks[empty_frame] = tracks[track_id].masks[nearest_filled_frame]
            tracks_in_frame[empty_frame] = track_ids
            filled_timesteps = np.array(sorted(list(tracks_in_frame.keys())))
    return tracks


if __name__ == '__main__':
    from skimage.morphology import disk
    from tracker.tracking import CellTrack
    from config import get_results_path
    DUMMY_TRACK1 = CellTrack(1)
    DUMMY_TRACK2 = CellTrack(2)
    DUMMY_TRACK3 = CellTrack(3)

    DUMMY_TRACK1.masks[0] = tuple(np.array(np.where(disk(10))) + np.array([15, 15] + np.array([20, 20])).reshape(-1, 1))
    DUMMY_TRACK2.masks[0] = tuple(np.array(np.where(disk(20))) + np.array([15, 15] + np.array([10, 10])).reshape(-1, 1))
    DUMMY_TRACK3.masks[0] = tuple(np.array(np.where(disk(30))) + np.array([15, 15]).reshape(-1, 1))
    ALL_TRACKS = {1: DUMMY_TRACK1, 2: DUMMY_TRACK2, 3: DUMMY_TRACK3}
    IMG_SHAPE = (100, 100)
    MASK = np.zeros(IMG_SHAPE)

    for k, v in ALL_TRACKS.items():
        MASK[v.masks[0]] += 1

    EXPORT = ExportResults()
    EXPORT(ALL_TRACKS, get_results_path() / 'dummyUnmerge', IMG_SHAPE, time_steps=[0])
