"""Utilities to track objects"""
import numpy as np
from tifffile import imread

from tracker.extract_data import get_indices_pandas
from tracker.flow import compute_fft_displacement
from tracker.graph import graph_tracking
from tracker.utils import compute_seeds

np.random.seed(42)


class MultiCellTracker:
    """Tracks multiple potentially splitting objects over time."""
    def __init__(self, config):
        """
        Initialises multi object tracker.
        Args:
            config: an instance of type TrackingConfig containing the parametrisation for the tracking algorithm
        """
        self.config = config
        self.cell_rois = {}
        self.tracklets = {}
        self.tracks = {}
        self.last_assigned_object = {}
        self.mapping_objects_tracks = {}
        self.segmentation_masks = {}
        self.img_shape = None
        self.delta_t = self.config.delta_t

    def __call__(self):
        """Tracks objects in the provided image data set"""
        time_steps = self.config.time_steps
        assert self.delta_t < len(time_steps), 'delta t larger than overall sequence length'
        assert self.delta_t < 10, 'max time span is 10'
        for i, time in enumerate(time_steps):
            print('#'*20)
            print('time point:', time)
            print('#'*20)
            self.tracking_step(i, time)
        # final matching step
        if (len(time_steps) - 1) % self.delta_t:
            self.matching_step()

        return self.tracks

    def tracking_step(self, i, time):
        """Applies a single tracking step."""
        self.propagate_tracklets(time)
        if (i % self.delta_t == 0) & (i > 0):
            self.matching_step()  # match objects
            # remove tracklets
            tracklets_to_pop = set(self.tracklets.keys()).difference(self.last_assigned_object.values())
            for tracklet_id in tracklets_to_pop:
                self.tracklets.pop(tracklet_id)
            succ_tracklets = list(filter(lambda x: len(self.tracks[self.mapping_objects_tracks[x]].successors) > 0,
                                         self.tracklets.keys()))
            old_tracklets = list(filter(lambda x: x[0] <= time - self.delta_t,
                                        self.tracklets.keys()))
            tracklets_to_remove = old_tracklets
            tracklets_to_remove.extend(succ_tracklets)

            for tracklet_id in set(tracklets_to_remove):
                self.tracklets.pop(tracklet_id)
            # remove matching candidates
            for tracklet_id in self.tracklets.keys():
                self.tracklets[tracklet_id].matching_candidates = set()

    def propagate_tracklets(self, time):
        """Propagates object position and features over time."""
        image = imread(self.config.get_image_file(time))
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

    def matching_step(self):
        """Matches tracks from previous time points to objects at current time point."""
        potential_matching_candidates = {t_id: tracklet.matching_candidates
                                         for t_id, tracklet in self.tracklets.items()}
        tracklet_features = {t_id: tracklet.all_features()
                             for t_id, tracklet in self.tracklets.items()}

        # match objects using coupled min cost flow
        # matches: segm_id: ([matched track_ids], is_cell_division)
        ad_costs = self.config.cut_off_distance
        matches = graph_tracking(tracklet_features, potential_matching_candidates,
                                 self.img_shape,
                                 cutoff_distance=ad_costs,
                                 allow_cell_division=self.config.allow_cell_division)

        # add selected objects to tracks
        for match in matches.values():
            # case no predecessor
            if match['pred_track_id']:
                # split case/ merge case
                self.init_new_track(match)
            else:
                # new track
                segm_id = tuple([int(el) for el in match['track'][0].split('_')])
                if segm_id not in self.mapping_objects_tracks.keys():
                    self.init_new_track(match)
                # append track
                else:
                    track_id = self.mapping_objects_tracks[segm_id]
                    self.append_to_track(track_id, match)
        # update last assigned object
        self.last_assigned_object = {t_id: track.last_assigned_obj
                                     for t_id, track in self.tracks.items()}

    def init_new_track(self, mapped_objects):
        """
        Creates a new track.
        Args:
            mapped_objects: dict contains predecessor/ successor information and
            which segmented objects have been assigned to track
        """
        if self.tracks.keys():
            track_id = max(list(self.tracks.keys())) + 1
        else:
            track_id = 1  # start with 1 as background is 0, no predecessor is 0
        if mapped_objects['pred_track_id']:
            pred_ids = [tuple([int(e) for e in el.split('_')])
                        for el in mapped_objects['predecessor']]
            pred_track_id = [self.mapping_objects_tracks[pred] for pred in pred_ids
                             if pred in self.mapping_objects_tracks]

            for t_id in pred_track_id:
                self.tracks[t_id].successors.add(track_id)
        else:
            pred_track_id = [0]
        if pred_track_id:
            self.tracks[track_id] = CellTrack(track_id, pred_track_id)
            self.append_to_track(track_id, mapped_objects)

    def append_to_track(self, track_id, mapped_objects):
        """
        Appends set of segmented objects to an existing track.
        Args:
            track_id: int providing the tracking id the segmented objects shall be assigned to
            mapped_objects: dict contains predecessor/ successor information and
            which segmented objects have been assigned to track
        """
        tracklet_ids = [tuple([int(e) for e in el.split('_')]) for el in mapped_objects['track']]
        for tracklet_id in tracklet_ids:
            self.mapping_objects_tracks[tracklet_id] = track_id
            self.tracks[track_id].add_time_step(tracklet_id, self.tracklets[tracklet_id].mask)


class CellTrack:
    """Contains the information of a single cell track"""
    def __init__(self, track_id, pred_track_id=[0]):
        self.track_id = track_id
        self.pred_track_id = pred_track_id
        self.successors = set()
        self.masks = {}
        self.last_assigned_obj = None

    def add_time_step(self, mask_id, mask_indices):
        """
        Add segmented object to the track.
        Args:
            mask_id: tuple providing the time point and the segmentation mask index
            mask_indices: tuple of segmentation mask pixel positions
        """
        time, _ = mask_id
        self.masks[time] = mask_indices
        self.last_assigned_obj = mask_id

    def get_last_position(self):
        last_time = sorted(list(self.masks.keys()))[-1]
        return np.median(np.stack(self.masks[last_time]), axis=-1)

    def get_first_position(self):
        first_time = sorted(list(self.masks.keys()))[0]
        return np.median(np.stack(self.masks[first_time]), axis=-1)

    def get_last_time(self):
        return sorted(list(self.masks.keys()))[-1]


class Tracklet:
    """Contains the information of a tracklet, which is a segmented object which position is
     propagated coarsely over time."""
    def __init__(self, tracklet_id, t_start, init_mask, roi_size, delta_t):
        """
        Initialises a tracklet.
        Args:
            tracklet_id: int indicating unique tracklet id
            t_start: int indicating start time point
            init_mask: tuple of initial segmentation mask pixels
            roi_size: tuple providing the ROI size
            delta_t: int providing the maximal temporal length of tracklet
        """
        self.tracklet_id = tracklet_id
        self.t_start = t_start
        self.mask = init_mask
        self.estimated_mask = {t_start: self.mask}
        self.delta_t = delta_t
        box_shape = (np.min(np.array(init_mask), axis=-1), np.max(np.array(init_mask), axis=-1))
        self.init_features = (box_shape, compute_seeds(init_mask), np.median(init_mask, axis=-1))
        self.matching_candidates = set()
        self.init_position = np.median(init_mask, axis=-1)
        self.position_estimate = {}
        self.estimated_features = {}
        self.features = {}
        self.init_size = None
        # calc own roi size
        self.roi = CellROI(self.init_position, self.calc_roi_size(roi_size))
        self.time_steps = []

    def calc_roi_size(self, roi_size_default):
        upper_left = np.min(self.mask, axis=-1)
        lower_right = np.max(self.mask, axis=-1)
        diff = lower_right - upper_left + 1
        diff = 2 * diff
        diff = np.max(np.stack([diff, roi_size_default]), axis=0)
        return diff

    def init_img_patch(self, time, image):
        self.roi(time, image)
        self.time_steps.append(time)

    def propagate(self, time, image):
        """Propagates ROI and positional features over time."""
        if (time - self.t_start) <= self.delta_t:
            self.roi(time, image)
            self.time_steps.append(time)
            if len(self.position_estimate) > 0:
                prev_position = self.position_estimate[sorted(self.position_estimate.keys())[-1]]
            else:
                prev_position = self.init_position
            last_mask = self.estimated_mask[max(list(self.estimated_mask.keys()))]
            img_border = np.array(image.shape).reshape(-1, 1) -1
            mask_estimate = np.array(last_mask) + self.roi.get_last_displacement().reshape(-1, 1)
            mask_estimate[mask_estimate < 0] = 0
            mask_estimate[mask_estimate > img_border] = np.tile(img_border,
                                                                (1, mask_estimate.shape[-1]))[mask_estimate > img_border]
            self.estimated_mask[time] = tuple(mask_estimate.astype(np.int32))
            # last features
            if self.estimated_features:
                last_estimate = self.estimated_features[max(self.estimated_features.keys())]
            else:
                last_estimate = self.init_features

            self.position_estimate[time] = prev_position + self.roi.get_last_displacement()
            # (2, n_dims) = box_shape dimensions
            box_shape = np.array([np.min(np.array(self.estimated_mask[time]), axis=-1),
                                  np.max(np.array(self.estimated_mask[time]), axis=-1)]).T
            box_shape[box_shape < 0] = 0
            box_shape[box_shape > img_border] = np.tile(img_border, (1, box_shape.shape[-1]))[box_shape > img_border]

            last_seeds = np.array(last_estimate[1]).astype(np.uint32)
            seeds = (last_seeds + self.roi.get_last_displacement().reshape(-1, 1)).astype(np.uint32)
            seeds[seeds < 0] = 0
            seeds[seeds > img_border] = np.tile(img_border, (1, seeds.shape[-1]))[seeds > img_border]

            seeds = tuple(seeds)
            if self.estimated_features:
                current_estimate = self.estimated_features[max(self.estimated_features.keys())]
            else:
                current_estimate = self.init_features
            self.features[time] = (*current_estimate, self.position_estimate[time])
            self.estimated_features[time] = (tuple(box_shape.T), seeds, self.position_estimate[time])

    def all_features(self):
        init_feat = (*self.init_features, self.init_features[-1])
        return init_feat, self.features

    def add_matching_candidates(self, n_ids):
        self.matching_candidates.update(n_ids)


class CellROI:
    """Defines region of interests (ROI) for a segmented object."""
    def __init__(self, init_position, roi_size):
        """
        Initialises the ROI of a segmented object.
        Args:
            init_position: tuple of the initial object position
            roi_size: tuple providing the ROI size
        """
        self.init_position = init_position
        self.roi_size = roi_size
        self.roi = {}  # time_point: ROI (center,box size)
        self.last_img_patch = None
        self.active = True
        self.displacement = {}

    def __call__(self, time, image):
        self._propagate(time, image)

    def _propagate(self, time, image):
        """Propagates ROI."""
        if self.roi:
            last_roi = self.last_roi()
            img_patch = image[last_roi.crop_box(image.shape)]
            # compute shift between the image patches
            # movement model: p_1 = p_0 + displacement -> new ROI position
            self.displacement[time] = compute_displacement(self.last_img_patch, img_patch)
            new_center_pos = compute_center_position(last_roi.center, self.displacement[time])
        else:
            new_center_pos = self.init_position
        new_center_pos[new_center_pos < 0] = 0
        new_center_pos[new_center_pos > image.shape] = np.array(image.shape)[new_center_pos > image.shape]
        new_roi = self.create_roi(new_center_pos)
        self.roi[time] = new_roi
        self.last_img_patch = image[new_roi.crop_box(image.shape)].copy() # avoid references on large image

    def get_last_displacement(self):
        keys = sorted(list(self.roi.keys()))
        return self.displacement[keys[-1]]

    def last_roi(self):
        keys = sorted(list(self.roi.keys()))
        return self.roi[keys[-1]]

    def last_roi_crop_box(self, img_shape):
        keys = sorted(list(self.roi.keys()))
        return self.roi[keys[-1]].crop_box(img_shape)

    def create_roi(self, center_point):
        return ROI(center_point, self.roi_size)


def compute_displacement(patch_1, patch_2):
    """Computes the shift between 2 images"""
    assert patch_1.shape == patch_2.shape, 'mismatching shapes'
    displacement = compute_fft_displacement(patch_1, patch_2)
    return displacement


def compute_center_position(old_position, displacement):
    return old_position + displacement


class TrackingConfig:
    """Contains the tracker configuration"""
    def __init__(self, img_files, segm_files, roi_box_size, delta_t=2,
                 cut_off_distance=None, allow_cell_division=True):
        """

        Args:
            img_files: a dict with time point: raw img file path
            segm_files: a dict with time point: segmentation file path
            roi_box_size: a tuple providing the default size of the ROI
            delta_t: maximum temporal length of tracklets
            cut_off_distance: a tuple providing a distance threshold for objects to appear/ disappear
            allow_cell_division: a boolean indicating whether to consider splitting objects
        """
        self.img_files = img_files
        self.segm_files = segm_files
        self.time_steps = sorted(list(self.img_files.keys()))
        self.roi_box_size = roi_box_size
        self.cut_off_distance = cut_off_distance
        self.delta_t = delta_t
        self.allow_cell_division = allow_cell_division
        if self.cut_off_distance is None:
            self.cut_off_distance = max(roi_box_size) / 2

    def get_image_file(self, time_step):
        return self.img_files[time_step]

    def get_segmentation_masks(self, time_step):
        segmentation = imread(self.segm_files[time_step])
        segmentation = np.squeeze(segmentation)
        return segmentation, get_indices_pandas(segmentation)


class ROI:
    """Defines a single region of interest (ROI)."""
    def __init__(self, center_position, box_size):
        """

        Args:
            center_position: init_position: tuple of the initial object position
            box_size: tuple providing the ROI size
        """
        self.center = np.array(center_position)
        self.box_size = tuple(box_size)
        self.top_left = np.array(np.array(center_position) - np.array(box_size) // 2, np.int)
        self.bottom_right = np.array(np.array(center_position) + np.array(box_size) // 2
                                     + np.array(box_size) % 2, np.int)

    def crop_box(self, img_shape):
        # this is basically a[0]:b[0],a[1]:b[1],... to select a set of indices from an array
        return tuple([slice(min(max(0, a), s), min(max(0, b), s))
                      for a, b, s in zip(self.top_left, self.bottom_right, img_shape)])

    def adapt_roi(self, center):
        self.center = np.array(center)
        self.top_left = np.array(np.array(self.center) - np.array(self.box_size) // 2, np.int)
        self.bottom_right = np.array(np.array(self.center) + np.array(self.box_size) // 2 +
                                     np.array(self.box_size) % 2,
                                     np.int)
