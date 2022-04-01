"""Utitlites to reproduce synthetically degraded segmentation mask
with n% missing masks (false negavices), under- and over-segmented objects."""
import os
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from scipy.ndimage.morphology import binary_closing, binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import ball, disk
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
from tifffile import imread, imsave

from tracker.extract_data import get_img_files, get_indices_pandas
from tracker.utils import collect_leaf_paths

np.random.seed(42)  # make mask edits reproducible


def extract_segmentation_data(data_path):
    """Extracts al segmentation masks from a sequence of images."""
    images = get_img_files(data_path)
    segmentation_masks = {t: get_indices_pandas(imread(img))
                          for t, img in images.items()}
    img_size = imread(images[list(images.keys())[0]]).shape
    return segmentation_masks, img_size


def get_masks(segm_masks):
    return [(time, m_id)
            for time, masks in segm_masks.items()
            for m_id in masks.keys()]


def remove_masks(segm_masks, remove_n_percent, return_edit_masks=False):
    """
    Models false negatives by randomly removing n% of segmentation masks.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        remove_n_percent: float indicating the fraction of segmentation masks to remove
        return_edit_masks: boolean, if true return the added masks, edited mask as well

    Returns: segmentation masks, (removed masks, new masks)

    """
    mask_ids = get_masks(segm_masks)
    select_n_masks = int(np.rint(remove_n_percent / 100 * len(mask_ids)))
    selected_ids = np.random.choice(len(mask_ids), select_n_masks, replace=False)

    for s_id in selected_ids:
        time, m_id = mask_ids[s_id]
        segm_masks[time].pop(m_id)
    if return_edit_masks:
        removed_masks = [mask_ids[s_id] for s_id in selected_ids]
        new_masks = []
        return segm_masks, removed_masks, new_masks
    return segm_masks


def merge_masks(segm_masks, merge_n_percent, return_edit_masks=False):
    """
    Models under-segmentation errors by randomly merging n% of segmentation masks.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        merge_n_percent: float indicating the fraction of segmentation masks to merge
        return_edit_masks: boolean, if true return the edited masks as well

    Returns: segmentation masks, (merged masks, new masks)

    """
    # define pairs of segmentation masks that can be merged based
    # on their distance to each other
    initial_keys = {k: list(v.keys()) for k, v in segm_masks.items()}
    initial_masks = get_masks(segm_masks)
    possible_merge_pairs = {}  #(time, m_id): {(time, m_id) : cost ,....
    merged_masks = []
    new_masks = []
    all_centroids = {(t, m_id): np.mean(segm_masks[t][m_id], axis=-1) for t, m_id in initial_masks}
    for t, m_id in initial_masks:
        mask_centroid = all_centroids[(t, m_id)]
        possible_merge_pairs[(t, m_id)] = {(t, other_id): np.linalg.norm(mask_centroid-all_centroids[(t, other_id)])
                                           for other_id, other_mask in segm_masks[t].items()
                                           if other_id != m_id}

    select_n_masks = np.rint(merge_n_percent / 100 * len(initial_masks))

    # segmented objects are merged iteratively -> more than two objects
    # can be merged into a single object -> keep distance info to other masks of
    # original segmentation masks and only update their segmentation masks
    while (len(merged_masks) < select_n_masks) and possible_merge_pairs:
        # sample per track its best merge partner - based on the distance
        potential_merge = {}
        for mask_id, neighbors in possible_merge_pairs.items():
            other_masks, distances = list(zip(*[(k, v) for k, v in neighbors.items()]))
            min_distance = min(distances)
            closest_mask = other_masks[distances.index(min_distance)]
            potential_merge[((mask_id, closest_mask))] = min_distance + 1

        # select closest 100 pairs and select one of these randomly
        pairs = [(k, v) for k, v in potential_merge.items()]
        pairs.sort(key=lambda x: x[1])
        distance_centroids = np.array([pair[1] for pair in pairs])
        index = np.argmax(distance_centroids > np.median(distance_centroids))
        pairs = pairs[:max(index, 100)]

        merge_pairs, dist = list(zip(*pairs))
        sampling_weight = 1 / np.array(dist)
        sampling_weight = np.array(sampling_weight) / np.sum(sampling_weight)

        selected_index = np.random.choice(len(merge_pairs), size=1, p=sampling_weight)[0]
        selected_masks = merge_pairs[selected_index]

        # merge masks
        time = selected_masks[0][0]
        merged_mask_id = max(max(segm_masks[time].keys()), max(initial_keys[time])) + 1
        mask_1, mask_2 = [np.array(segm_masks[t][m_id]) for t, m_id in selected_masks]
        segm_masks[time][merged_mask_id] = compute_merged_mask(mask_1, mask_2)
        new_masks.append((time, merged_mask_id))

        for mask_id in selected_masks:
            # only append to merged masks if in initial segmentation masks,
            # as in total n% of initial masks should be merged in the end
            if mask_id in possible_merge_pairs:
                possible_merge_pairs.pop(mask_id)
                merged_masks.append(mask_id)
            # remove mask as their indices are already added in merged mask
            segm_masks[mask_id[0]].pop(mask_id[1])
            if mask_id in new_masks:
                new_masks.remove(mask_id)
            for k in possible_merge_pairs.keys():
                if mask_id in possible_merge_pairs[k]:
                    other_distance = possible_merge_pairs[k].pop(mask_id)
                    if (time, merged_mask_id) in possible_merge_pairs[k]:
                        #  same key but multiple distances - select min distance
                        possible_merge_pairs[k][(time, merged_mask_id)] = min(possible_merge_pairs[k][(time, merged_mask_id)],
                                                                              other_distance)
                    else:
                        possible_merge_pairs[k][(time, merged_mask_id)] = other_distance
    if return_edit_masks:
        return segm_masks, merged_masks, new_masks
    else:
        return segm_masks


def compute_merged_mask(mask_1, mask_2):
    """Merges two segmentation masks."""
    merged_object_ind = np.hstack([mask_1, mask_2])

    max_dist = int(np.ceil(calc_hull_min_distance(mask_1, mask_2)) + 1)

    box_shape = np.max(merged_object_ind, axis=1) - np.min(merged_object_ind, axis=1) \
                + 2*max_dist + 1  # add background border
    original_masks = np.zeros(tuple(box_shape))
    original_masks[tuple(merged_object_ind - np.min(merged_object_ind, axis=1).reshape(-1, 1) + max_dist)] = 1

    # for large distances dilation to memory intense -> down sample large blobs and rescale back
    orig_size = original_masks.shape
    max_s = 200
    if max(orig_size) > max_s:
        r = max(orig_size) / max_s
        scales = np.array(orig_size) / r
        scales[r < 1] = 1
        rescale_shape = np.ceil(scales).astype(np.uint32)
        original_masks = resize(original_masks, rescale_shape)
        max_dist = int(np.ceil(max_dist / r))

    if (max_dist > 20) and (len(original_masks.shape) == 3):
        r = max_dist / 20
        scales = np.array(original_masks.shape) / r
        scales[r < 1] = 1
        rescale_shape = np.ceil(scales).astype(np.uint32)
        original_masks = resize(original_masks, rescale_shape)
        max_dist = int(np.ceil(max_dist / r))

    if len(box_shape) == 2:
        structure = disk(max_dist)
    elif len(box_shape) == 3:
        structure = ball(max_dist)
    else:
        raise AssertionError(f'Input masks are not 2D or 3D {original_masks.shape}')
    merged_mask = binary_closing(original_masks, structure)
    if merged_mask.shape != orig_size:
        merged_mask = resize(merged_mask, orig_size)
        # fill holes in orig masks due to downsampling
        merged_mask[tuple(merged_object_ind - np.min(merged_object_ind, axis=1).reshape(-1, 1) + max_dist)] = 1

    merged_object_ind = np.array(np.where(merged_mask)) + np.min(merged_object_ind, axis=1).reshape(-1, 1) - max_dist
    return tuple(merged_object_ind)


def calc_min_distance(mask_1, mask_2):
    """Calculates the minimum distance between two masks."""
    mask_1 = np.array(mask_1)
    mask_2 = np.array(mask_2)
    d = mask_1.transpose()[..., np.newaxis] - mask_2[np.newaxis, ...]
    min_dist = np.min(np.sum(d**2, axis=1)**0.5)
    return min_dist


def calc_hull_min_distance(mask_1, mask_2):
    """Calculates the minimum distance between the hulls of two masks."""
    hull_width = 1
    mask_1 = np.array(mask_1)
    mask_2 = np.array(mask_2)
    dummy_1 = np.zeros(tuple(np.max(mask_1, axis=1) - np.min(mask_1, axis=1) + 2*hull_width +1))
    dummy_1[tuple(mask_1 - np.min(mask_1, axis=1).reshape(-1, 1) + hull_width)] = 1

    dummy_2 = np.zeros(tuple(np.max(mask_2, axis=1) - np.min(mask_2, axis=1) + 2*hull_width +1))
    dummy_2[tuple(mask_2 - np.min(mask_2, axis=1).reshape(-1, 1) + hull_width)] = 1

    if len(dummy_1.shape) == 2:
        structure = disk(hull_width)
    elif len(dummy_1.shape) == 3:
        structure = ball(hull_width)
    else:
        raise AssertionError(f'Input masks are not 2D or 3D {dummy_1.shape}')
    hull_1 = dummy_1 - binary_erosion(dummy_1, structure)
    hull_2 = dummy_2 - binary_erosion(dummy_2, structure)

    hull_indices_1 = np.array(np.where(hull_1)) + np.min(mask_1, axis=1).reshape(-1, 1) - hull_width
    hull_indices_2 = np.array(np.where(hull_2)) + np.min(mask_2, axis=1).reshape(-1, 1) - hull_width
    return calc_min_distance(hull_indices_1, hull_indices_2)


def compute_seeds(mask, n_seeds):
    """Computes seed points to split a segmentation mask."""
    mask = np.array(mask)

    box_shape = np.max(mask, axis=1) - np.min(mask, axis=1) + 3  # add background border
    dummy = np.zeros(tuple(box_shape))
    dummy[tuple(mask - np.min(mask, axis=1).reshape(-1, 1) + 1)] = 1
    dist = distance_transform_edt(dummy)
    stacked = np.stack(np.gradient(dist))
    abs_grad = np.sum(stacked**2, axis=0)
    seed_points = np.where((abs_grad < 0.1) & (dist > 0))
    if len(seed_points[0]) < n_seeds:
        seed_points = tuple(mask)
    else:
        # compute non shifted position
        seed_points = np.array(seed_points) + np.min(mask, axis=1).reshape(-1, 1) - 1
    seed_index = np.random.choice(len(seed_points[0]), n_seeds, replace=False)
    seed_points = np.array(seed_points)[..., seed_index]
    return seed_points


def split_masks(segm_masks, split_n_percent, return_edit_masks=False):
    """
    Models over-segmentation errors by randomly splitting n% of segmentation masks.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        split_n_percent: float indicating the fraction of segmentation masks to split
        return_edit_masks: boolean, if true return the edited masks as well

    Returns: segmentation masks, (merged masks, new masks)


    """
    mask_ids = get_masks(segm_masks)
    # skip too small masks :n mask pixels <5
    mask_ids = [m for m in mask_ids if len(segm_masks[m[0]][m[1]][0]) >= 5]

    select_n_masks = int(np.rint(split_n_percent / 100 * len(mask_ids)))
    selected_ids = np.random.choice(len(mask_ids), select_n_masks, replace=False)
    new_masks = []
    for s_id in selected_ids:
        time, m_id = mask_ids[s_id]
        mask = np.array(segm_masks[time].pop(m_id))
        seed_positions = compute_seeds(mask, 2)
        nn = NearestNeighbors(n_neighbors=1).fit(seed_positions.T)
        _, indices = nn.kneighbors(mask.T)
        mask_id_max = max(segm_masks[time].keys())
        for i, _ in enumerate(seed_positions):
            segm_masks[time][mask_id_max + i + 1] = tuple(mask.T[indices.reshape(-1) == i].T)
            new_masks.append((time, mask_id_max + i + 1))
    if return_edit_masks:
        splitted_masks = [mask_ids[s_id] for s_id in selected_ids]
        return segm_masks, splitted_masks, new_masks
    else:
        return segm_masks


def export_masks(segm_masks, export_path, img_size):
    """Exports segmentation mask to sequences of segmentation images."""
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    t_max = max(3, np.ceil(np.log10(max(segm_masks.keys()))))
    for t, masks in segm_masks.items():
        img = np.zeros(img_size, dtype=np.uint16)
        for m_id, mask_indices in masks.items():
            mask_indices = np.array(mask_indices)
            valid_index = mask_indices < np.array(img_size).reshape(-1, 1)
            mask_indices = tuple(mask_indices[:, np.all(valid_index, axis=0)])
            img[mask_indices] = m_id

        img_name = 'mask' + str(t).zfill(t_max) + '.tif'
        imsave(export_path / img_name, img.astype(np.uint16), compress=2)


def combine_segm_errors(segm_masks, n_percent_errors):
    """
    Models a combination of errors by randomly editing of segmentation masks:
     remove n%/3 randomly (false negatives), merge n%/3 randomly (under-segmentation errors),
     split n%/3 randomly (over-segmentation errors).
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        n_percent_errors: float indicating the fraction of segmentation masks to edit

    Returns: segmentation masks

    """
    final_masks = {}
    n_masks_initial = len(get_masks(segm_masks))
    # split error percentage evenly over split, merge, remove errors
    for i in range(3):

        n_masks_remaining = len(get_masks(segm_masks))
        n_percent_faction = n_masks_initial / n_masks_remaining * n_percent_errors / 3
        if i == 0:
            segm_masks, edited_masks_ids, new_mask_ids = merge_masks(segm_masks,
                                                                     n_percent_faction,
                                                                     return_edit_masks=True)
        elif i == 1:
            segm_masks, edited_masks_ids, new_mask_ids = split_masks(segm_masks,
                                                                     n_percent_faction,
                                                                     return_edit_masks=True)

        else:
            segm_masks, edited_masks_ids, new_mask_ids = remove_masks(segm_masks,
                                                                      n_percent_faction,
                                                                      return_edit_masks=True)
        # masks already removed by error functions as dict with data frame as values and this is a call by reference
        for m_id in new_mask_ids:
            time, mask_id = m_id
            if time not in final_masks:
                final_masks[time] = {}
            # in merge multi merge possible -> new ids can get overwritten
            final_masks[time][len(final_masks[time])+1] = segm_masks[time].pop(mask_id)

    for k, v in segm_masks.items():
        if k not in final_masks:
            final_masks[k] = {}
        if len(final_masks[k]) > 0:
            max_id = max(list(final_masks[k].keys()))
        else:
            max_id = 0
        final_masks[k].update({(max_id + i + 1): vv[1] for i, vv in enumerate(v.items())})
    return final_masks


def set_up_synth_segm_errors(data_path, results_path, percentages, n_runs):
    """
    Creates synthetically degraded segmentation results.
    Args:
        data_path: path to ground truth segmentation images
        results_path: path where to save segmentation images with added errors
        percentages: list containing percentages of segmentation errors
        n_runs: int indicating the number of experiments to run per error type
                and segmentation error percentage
    """
    for i in range(1, 3):
        print(data_path.name, str(i).zfill(2))
        d_path = data_path / (str(i).zfill(2) + '_GT') / 'SEG'
        for n_percent in percentages:
            print(f'edit {n_percent}%')
            for n in range(n_runs):
                print(f'run {n+1}/{n_runs}')
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks = combine_segm_errors(segm_masks, n_percent)
                export_path = results_path / str(i).zfill(2) / 'mixed' / ('percentage_' + str(n_percent)) / (
                        'run_' + str(n))
                export_masks(segm_masks, export_path, img_size)

                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks = remove_masks(segm_masks, n_percent)
                export_path = results_path / str(i).zfill(2) / 'remove' / ('percentage_' + str(n_percent)) / ('run_' + str(n))
                export_masks(segm_masks, export_path, img_size)

                segm_masks = split_masks(extract_segmentation_data(d_path)[0], n_percent)
                export_path = results_path / str(i).zfill(2) / 'split' / ('percentage_' + str(n_percent)) / ('run_' + str(n))
                export_masks(segm_masks, export_path, img_size)

                segm_masks = merge_masks(extract_segmentation_data(d_path)[0], n_percent)
                export_path = results_path / str(i).zfill(2) / 'merge' / ('percentage_' + str(n_percent)) / ('run_' + str(n))
                export_masks(segm_masks, export_path, img_size)


def visualize_runs(data_path):
    """
    Visualizes different runs of an experiment.
    Args:
        data_path: path to root dir ..error_type/percentage_x/
    """
    all_paths = collect_leaf_paths(data_path)
    img_f = get_img_files(all_paths[0])
    for f in img_f.values():
        _, ax = plt.subplots(1, len(all_paths))
        for i, p in enumerate(all_paths):
            img = imread((p / Path(f).name).as_posix())
            ax[i].imshow(img)
        plt.gcf().suptitle(Path(f).name)
        plt.show()


def visualize_error_masks(img_path, segm_path, percentage):
    """
   Visualizes raw images, ground truth data and simulated segmentation errors.
    Args:
        img_path: path to the original image data set
        segm_path: path containing all synth error results
        percentage: int to select data sets with n% segmentation errors to visualize

    """
    all_paths = collect_leaf_paths(segm_path)
    # (error_type, percentage, run)
    data_paths = {(p.parts[-3],
                   float(p.parts[-2].split('_')[-1]),
                   int(p.parts[-1].split('_')[-1])): p for p in all_paths}
    data_paths = pd.DataFrame.from_dict(data_paths, orient='index')
    data_info = list(zip(*data_paths.index.values))
    data_paths['path'] = data_paths[0]
    data_paths['error'] = data_info[0]
    data_paths['percentage'] = data_info[1]
    data_paths['run'] = data_info[2]
    data_paths = data_paths.reset_index()
    data_paths = data_paths.groupby(['percentage'])
    data_paths = data_paths.get_group(percentage)
    img_files = get_img_files(img_path)
    gt_path = img_path.parent / (img_path.name+'_GT') / 'TRA'
    cmap_mask = copy(plt.cm.tab10)
    cmap_mask.set_bad('k', 1)
    for time in sorted(img_files.keys()):
        img_f = Path(img_files[time])
        _, ax = plt.subplots(len(data_paths['run'].unique()), 1+len(data_paths['error'].unique()))
        ax[0][0].imshow(imread(img_f), cmap='gray')
        gt_masks = imread(gt_path/('man_track'+img_f.name[1:]))
        ax[1][0].imshow(np.ma.masked_array(gt_masks, mask=gt_masks == 0), cmap=cmap_mask)
        ax[0][0].set_ylabel('raw_img')
        ax[1][0].set_ylabel('gt')

        for i in range(len(ax[0])):
            ax[i][0].xaxis.set_ticks([])
            ax[i][0].yaxis.set_ticks([])
            if i > 1:
                ax[i][0].axis('off')
        for i_row, error_group in enumerate(data_paths.groupby(['run'])):
            for j_col, d in enumerate(error_group[1].iterrows()):
                segm_mask = imread(d[1]['path'] / ('mask'+img_f.name[1:]))
                ax[i_row][j_col+1].imshow(np.ma.masked_array(segm_mask, mask=segm_mask == 0), cmap=cmap_mask)
                if j_col == 1:
                    ax[i_row][j_col].set_ylabel('run ' + str(d[1]['run']))
                if i_row == len(ax[0])-1:
                    ax[i_row][j_col+1].set_xlabel(d[1]['error'])
                ax[i_row][j_col+1].xaxis.set_ticks([])
                ax[i_row][j_col+1].yaxis.set_ticks([])

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)
        plt.gcf().suptitle(Path(img_f).name)
        plt.show()


def extract_synth_data_paths(segm_path, percentage):
    """
    Extracts all data paths which have n% of modified segmentation masks
    Args:
        segm_path: path containing all synthetic error results
        percentage: int to select data sets with n% segmentation errors to visualize
    """
    all_paths = collect_leaf_paths(segm_path)
    # (error_type, percantage, run)
    data_paths = {(p.parts[-3],
                   float(p.parts[-2].split('_')[-1]),
                   int(p.parts[-1].split('_')[-1])): p for p in all_paths}
    data_paths = pd.DataFrame.from_dict(data_paths,
                                        orient='index')
    data_info = list(zip(*data_paths.index.values))
    data_paths['path'] = data_paths[0]
    data_paths['error'] = data_info[0]
    data_paths['percentage'] = data_info[1]
    data_paths['run'] = data_info[2]
    data_paths = data_paths.reset_index()
    data_paths = data_paths.groupby(['percentage'])
    data_paths = data_paths.get_group(percentage)
    return data_paths


def save_group(img_file, segm_path, res_path, percentage):
    """Save a selected group of images to another folder to create plot for publication."""
    # save all synth error files belonging to image
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    cmap_mask = copy(plt.cm.tab10)
    file_ending = 'png'
    data_paths = extract_synth_data_paths(segm_path, percentage)
    img_file = Path(img_file)
    plt.imshow(imread(img_file), cmap='gray')
    plt.axis('off')
    plt.savefig(res_path / ('raw_img' + '.' + file_ending), bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.axis('off')
    gt_img = imread(img_file.parent.parent / (img_file.parent.name + '_GT') / 'TRA' / ('man_track'+img_file.name[1:]))
    cmap_list = [cmap_mask(i % cmap_mask.N) for i in range(np.max(gt_img))]
    cmap_list = mpl_colors.LinearSegmentedColormap.from_list('custom', cmap_list, len(cmap_list))
    cmap_list.set_bad('k', 1)
    plt.imshow(np.ma.masked_array(gt_img, mask=gt_img == 0),
               cmap=cmap_list,
               norm=mpl_colors.BoundaryNorm(np.linspace(0, cmap_list.N, cmap_list.N+1), cmap_list.N))

    plt.savefig(res_path / ('gt_img' + '.' + file_ending), bbox_inches='tight', pad_inches=0)
    plt.close()

    for d in data_paths.iterrows():
        segm_mask = imread(d[1]['path'] / ('mask'+img_file.name[1:]))
        f = plt.figure()
        cmap_list = [cmap_mask(i % cmap_mask.N) for i in range(np.max(segm_mask))]
        cmap_list = mpl_colors.LinearSegmentedColormap.from_list('custom', cmap_list, len(cmap_list))
        cmap_list.set_bad('k', 1)
        plt.axis('off')
        plt.imshow(np.ma.masked_array(segm_mask, mask=segm_mask == 0),
                   cmap=cmap_list,
                   norm=mpl_colors.BoundaryNorm(np.linspace(0, cmap_list.N, cmap_list.N + 1), cmap_list.N))
        plt.savefig(res_path / (' '.join(['segmentation', d[1]['error'], str(d[1]['run'])]) + '.' + file_ending),
                    bbox_inches='tight', pad_inches=0)
        plt.close(f)


def vis_run(img_path, segm_paths):
    """Visualizes a selected set of segmentation results."""
    img_f = get_img_files(img_path)
    gt_path = img_path.parent / (img_path.name + '_GT') / 'TRA'
    cmap_mask = copy(plt.cm.tab10)

    for f in img_f.values():
        show_plot = False
        f = Path(f)
        img = imread(f)
        if len(img.shape) == 2:
            _, ax = plt.subplots(1, 2+len(segm_paths))
        elif len(img.shape) == 3:
            _, ax = plt.subplots(3, 2+len(segm_paths))
        else:
            raise AssertionError('unknown image size')

        gt_masks = imread(gt_path / ('man_track' + f.name[1:]))
        cmap_list = [cmap_mask(i % cmap_mask.N) for i in range(np.max(gt_masks))]
        cmap_list = mpl_colors.LinearSegmentedColormap.from_list('custom', cmap_list, len(cmap_list))
        cmap_list.set_bad('k', 1)
        if len(img.shape) == 2:
            ax[0].imshow(img, cmap='gray')
            ax[1].imshow(np.ma.masked_array(gt_masks, mask=gt_masks == 0), cmap=cmap_list,
                         norm=mpl_colors.BoundaryNorm(np.linspace(0, cmap_list.N, cmap_list.N + 1), cmap_list.N))

        else:
            for i_row in range(3):
                ax[i_row][0].imshow(np.max(img, axis=i_row), cmap='gray')
                ax[i_row][1].imshow(np.max(np.ma.masked_array(gt_masks, mask=gt_masks == 0), axis=i_row), cmap=cmap_list,
                                    norm=mpl_colors.BoundaryNorm(np.linspace(0, cmap_list.N, cmap_list.N + 1), cmap_list.N))
        for j_path, seg_p in enumerate(segm_paths):
            segm_mask = imread(seg_p / ('mask' + f.name[1:]))
            cmap_list = [cmap_mask(i % cmap_mask.N) for i in range(np.max(segm_mask))]
            cmap_list = mpl_colors.LinearSegmentedColormap.from_list('custom', cmap_list, len(cmap_list))
            cmap_list.set_bad('k', 1)
            if len(img.shape) == 2:
                ax[2 + j_path].imshow(np.ma.masked_array(segm_mask, mask=segm_mask == 0), cmap=cmap_list,
                                      norm=mpl_colors.BoundaryNorm(np.linspace(0, cmap_list.N, cmap_list.N + 1), cmap_list.N))
            else:
                for i_row in range(3):
                    ax[i_row][2 + j_path].imshow(np.max(np.ma.masked_array(segm_mask, mask=segm_mask == 0), axis=i_row),
                                                 cmap=cmap_list,
                                                 norm=mpl_colors.BoundaryNorm(np.linspace(0, cmap_list.N, cmap_list.N + 1), cmap_list.N))

        plt.gcf().suptitle(Path(f).name)
        plt.show()


if __name__ == '__main__':
    from config import get_results_path, get_data_path
    ALL_DATA_PATHS = [get_data_path() / 'Fluo-N2DH-SIM+',
                      get_data_path() / 'Fluo-N3DH-SIM+']
    VIS_IMG_PATH = get_data_path() / 'Fluo-N2DH-SIM+/01'
    VIS_SEGMENTATION_PATH = get_results_path() / 'synth_segm_errors/Fluo-N2DH-SIM+/01'
    VIS_PERCENTAGE = 10

    PERCENTAGES = [1, 2, 5, 10, 20]
    N_RUNS = 5
    CREATE_DATA = True
    VISUALIZE_RUN = True

    if CREATE_DATA:
        for p in ALL_DATA_PATHS:
            data_path = Path(p)
            data_set = data_path.name
            res_path = get_results_path() / 'synth_segm_errors' / data_set
            set_up_synth_segm_errors(data_path, res_path, PERCENTAGES, N_RUNS)
    if VISUALIZE_RUN:
        visualize_error_masks(VIS_IMG_PATH, VIS_SEGMENTATION_PATH, VIS_PERCENTAGE)
