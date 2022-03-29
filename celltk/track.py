import os
import warnings
from typing import Tuple

import numpy as np
import btrack
import btrack.utils as butils
import btrack.constants as bconstants
import skimage.measure as meas
import skimage.segmentation as segm
import scipy.spatial.distance as distance

from celltk.core.operation import BaseTracker
from celltk.utils._types import Image, Mask, Track
from celltk.utils.utils import ImageHelper, stdout_redirected
from celltk.utils.operation_utils import (lineage_to_track,
                                          match_labels_linear,
                                          voronoi_boundaries,
                                          get_binary_footprint,
                                          track_to_mask, track_to_lineage,
                                          label_by_parent,
                                          data_from_regionprops_table,
                                          paired_dot_distance)
from celltk.utils.metric_utils import total_intensity

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
        """Tracks objects by optimizing the area overlap
        from frame to frame.

        :param mask: Mask with the objects to be tracked uniquely labeled.
        :param voronoi_split: If True, creates a voronoi
            map from the objects in mask. Uses that map
            to keep objects separated in each frame. Slows
            down computation.

        :return: Track with objects linked

        TODO:
            - Multiple ways to improve this function
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
        """Uses watershed to track from objects in one frame to objects
        in the next. Useful for objects that grow in size, but don't move
        much, such as bacterial colonies.

        :param mask: Mask with the objects to be tracked uniquely labeled.
        :param connectivity: Determines the local neighborhood around a pixel.
            Defined as the number of orthogonal steps needed to reach a pixel.
        :param watershed_line: If True, a one-pixel wide line separates the
            regions obtained by the watershed algorithm. The line is labeled 0.
        :param keep_seeds: If True, seeds from previous frame are always
            kept. This is more robust for segmentation that is incomplete
            or fragmented in some frames. However, the mask will be
            monotonically increasing in size. Not appropriate for images
            with drift or for moving objects.

        :return: Track with objects linked
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
    def linear_tracking_w_properties(self,
                                     mask: Mask,
                                     ) -> Track:
        """Like simple_linear_tracker, but with the ability
        to specify specific properties to use in the cost
        function
        """
        pass

    @ImageHelper(by_frame=False)
    def detect_cell_division(self,
                             image: Image,
                             track: Track,
                             mask: Mask = None,
                             displacement_thres: float = 30,
                             frame_thres: int = 3,
                             mass_thres: float = 0.15,
                             dist_thres: float = 0.35,
                             dot_thres: float = -0.7,
                             angle_weight: float = 0.5
                             ) -> Track:
        """Detects cells that have divided based on location, size,
        and intensity information. Daughter cells are expected to
        be approximately half the total intensity of the mother cell,
        and to have the mother cell roughly in line and between them.

        NOTE:
            - Any Track passed to this function will have all other
              division events removed.

        :param image: Image with intensity information.
        :param track: Track of objects to detect division on.
        :param mask: Mask of objects to detect division on. Note
            that the objects must already be linked from frame
            to frame for the algorithm to work.
        :param displacement_thres: Maximum distance allowed from
            the centroid of a parent cell to the centroid of a
            daughter cell.
        :param frame_thres: Maximum number of frames from the division
            event to a daughter cell.
        :param mass_thres: Maximum error for total intensity of a
            daughter cell relataive to half of the mother cell's
            intensity.
        :param dist_thres: Maximum error allowed for location of
            mother cell relative to location of the daughter cells.
            Mother cell is expected to be equidistant from the
            daughter cells.
        :param dot_thres: Maximum value of the normalized dot
            product between the two vectors from each daughter
            cell to the mother cell. Essentially a measure
            of the angle between the daughter cells.
        :param angle_weight: Weight given to angle and distance information
            when assigning daughter cells. If multiple candidate daughters are
            found, they are assigned based on intensity information
            and angle information.

        :return: Track with objects linked and cell division marked
        """
        # First, convert track to mask if needed
        if track is not None:
            mask = track_to_mask(track)
        else:
            assert mask is not None
            track = mask.copy()

        # Find start and end of each trace
        # app_diapp = label, first_frame, last_frame, parent_label
        app_disapp = track_to_lineage(track)
        fr_min, fr_max = (0, mask.shape[1])

        # Remove cells that appeared in first frame or disappered in last
        after_first = app_disapp[:, 1] > fr_min
        before_last = app_disapp[:, 2] < fr_max
        app = app_disapp[after_first]  # possible daughters
        dis = app_disapp[before_last]  # possible parents

        # Now we need the intensities and locations for the app and dis frames
        app_fr = app[:, 1]  # second col is first frame of trace
        dis_fr = dis[:, 2]  # third col is last frame of trace
        rel_frames = np.unique(np.hstack([app_fr, dis_fr]))
        image_data = {fr:
                      meas.regionprops_table(mask[fr], image[fr],
                                             ['label', 'centroid'],
                                             extra_properties=[total_intensity]
                                             )
                      for fr in rel_frames}

        # Extract data from the regionprops
        app_xy = np.vstack(
            data_from_regionprops_table(image_data, 'centroid',
                                        app[:, 0], app[:, 1])
        )
        app_mass = np.vstack(
            data_from_regionprops_table(image_data, 'total_intensity',
                                        app[:, 0], app[:, 1])
        )
        dis_xy = np.vstack(
            data_from_regionprops_table(image_data, 'centroid',
                                        dis[:, 0], dis[:, 2])
        )
        dis_mass = np.vstack(
            data_from_regionprops_table(image_data, 'total_intensity',
                                        dis[:, 0], dis[:, 2])
        )

        # All matrices should be parent (dis) x daughter (app)
        # Calculate frame distance and mask
        app_fr, dis_fr = np.meshgrid(app_fr, dis_fr)
        # uint data type, so negative values > frame_thres
        fr_dist = (app_fr - dis_fr)
        # cannot appear and disappear in the same frame
        fr_mask = (fr_dist <= frame_thres) * (fr_dist > 0)

        # Calculate Euclidean distance between centroids and mask
        euclid_dist = distance.cdist(dis_xy, app_xy)
        euclid_mask = euclid_dist <= displacement_thres

        # Calculate mass (intensity) distance and mask
        # (parent - daughter) + 0.5 = 1 is the assumption
        app_mass, dis_mass = np.meshgrid(app_mass, dis_mass)
        mass_dist = ((app_mass - dis_mass).astype(np.int16) / dis_mass) + 0.5
        mass_mask = np.abs(mass_dist) <= mass_thres

        # Now start finding distances between cells
        binary_cost = fr_mask * euclid_mask * mass_mask
        nonzero_idx = np.nonzero(binary_cost.any(1))[0]

        # For each possible parent cell
        for par in nonzero_idx:
            # Save indices associated with possible daughters
            binrow = binary_cost[par]
            cand_dau_idx = np.nonzero(binrow)[0]

            cand_par_xy = dis_xy[par]
            cand_dau_xy = app_xy[binrow]

            # To start, assume that none of the daughters are matches
            binary_cost[par, :] = False
            # Don't match a single daughter to a parent
            if len(cand_dau_xy) > 1:
                # Get angle and distance eror
                dot, dist = paired_dot_distance(cand_par_xy, cand_dau_xy)
                candidates = (dot <= dot_thres) * (dist <= dist_thres)
                candidate_pairs = np.transpose(np.nonzero(candidates))

                # Each row is a possible pair of daughters
                if candidate_pairs.shape[0] == 1:
                    # Two matching cells have been found
                    binary_idx = cand_dau_idx[candidate_pairs.ravel()]
                    binary_cost[par, binary_idx] = True
                elif candidate_pairs.shape[0] > 1:
                    # For multiple possible pairs, calculate assoc. costs
                    costs = []
                    for cd in candidate_pairs:
                        d0 = cand_dau_idx[cd[0]]
                        d1 = cand_dau_idx[cd[1]]
                        # Error based on sum of intensities
                        mass_error = np.abs(
                            np.sum([mass_dist[par, d0],
                                    mass_dist[par, d1]])
                        )
                        # Error based on location of daughters and parent
                        angle_error = ((1 - np.abs(dot[cd[0], cd[1]])) +
                                       dist[cd[0], cd[1]])

                        costs.append((1 - angle_weight) * mass_error +
                                     angle_weight * angle_error)

                    # Select the minimum cost
                    best = np.argmin(np.abs(costs))
                    binary_idx = cand_dau_idx[candidate_pairs[best].ravel()]
                    binary_cost[par, binary_idx] = True

        # Remove daughters that have been assigned more than once
        binary_cost[:, binary_cost.sum(0) > 1] = False
        binary_cost[binary_cost.sum(1) == 1, :] = False

        # Construct a lineage to assign the daughters found
        dau_idx = np.nonzero(binary_cost.any(0))[0]
        new_lineage = np.zeros((len(dau_idx), 4), dtype=np.uint16)
        for n, dau in enumerate(dau_idx):
            new_lineage[n, :] = app[dau]
            new_lineage[n, -1] = np.argmax(binary_cost[:, dau])

        return lineage_to_track(mask, new_lineage)

    @ImageHelper(by_frame=False)
    def kit_sch_ge_tracker(self,
                           image: Image,
                           mask: Mask,
                           default_roi_size: int = 2,
                           delta_t: int = 3,
                           cut_off_distance: Tuple = None,
                           allow_cell_division: bool = True,
                           postprocessing_key: str = None,
                           ) -> Track:
        """Tree-based tracking algorithm. First creates small
        tracklets within delta_t frames, then finds a globally optimal
        solution for linking the tracklets to form full tracks. Has
        built-in cell detection and can also combine objects.

        NOTE:
            - Objects can change in this algorithm, so the output is
            not guaranteed to have the exact same objects as the input
            mask.

        NOTE:
            - A current Gurobi license is required to use this algorithm.
            Personal licenses are free for academics. Please see _________
            if you need help installing or using a license. If you run into
            issues, please open an issue on Github. ______________

        NOTE:
            - This tracking algorithm was developed by ____________
            and can be found at _______________.

        :param image: Image with intensity information
        :param mask: Mask with objects to be tracked
        :param default_roi_size: Size of the region to look for connecting
            objects. Set relative to the mean size of the objects. i.e. 2
            means a search area twice as large as the mean object.
        :param delta_t: Number of frames in each window for forming tracklets.
        :param cut_off_distance: Maximum distance between linked objects
        :param allow_cell_division: If True, attempt to locate and mark
            dividing cells.
        :param postprocessing_key: TODO. See KIT-Sch-GE documentation

        TODO:
            - Add citation for kit sch ge
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
        """Wrapper for btrack, a Bayesian-based tracking algorithm.
        Please see: https://github.com/quantumjot/BayesianTracker

        :param mask: Mask with objects to be segmented
        :param config_path: Path to configuration file. Must be JSON.
        :param update_method: Method to use when optimizing the solution.
            Options are 'exact' and 'approximate'. Use approximate if
            exact is taking too long or utilizing excessive resources.

        :return: Track with objects linked
        TODO:
            - Add citation.and expand documentation
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
