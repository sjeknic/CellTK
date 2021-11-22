from typing import Dict

import numpy as np
from skimage.morphology import dilation, remove_small_holes
from skimage.measure import label
from scipy.ndimage import (binary_dilation, distance_transform_edt)
import SimpleITK as sitk

from cellst.utils._types import Image, Mask, Track, Arr

# TODO: Add label by parent function


def remove_small_holes_keep_labels(image: np.ndarray,
                                   size: float
                                   ) -> np.ndarray:
    """
    Wrapper for skimage.morphology.remove_small_holes
    to keep the same labels on the images.

    TODO:
        - Confirm correct selem to use or make option
    """
    dilated = dilation(image, selem=np.ones((3, 3)))
    fill = remove_small_holes(image, area_threshold=size,
                              connectivity=2, in_place=False)
    return np.where(fill > 0, dilated, 0)


def gray_fill_holes_celltk(labels):
    """
    Direct copy from CellTK, trying to make a copy function above.
    """
    fil = sitk.GrayscaleFillholeImageFilter()
    filled = sitk.GetArrayFromImage(fil.Execute(sitk.GetImageFromArray(labels)))
    holes = label(filled != labels)
    for idx in np.unique(holes):
        if idx == 0:
            continue
        hole = holes == idx
        surrounding_values = labels[binary_dilation(hole) & ~hole]
        uniq = np.unique(surrounding_values)
        if len(uniq) == 1:
            labels[hole > 0] = uniq[0]
    return labels


def track_to_mask(track: Track, idx: np.ndarray = None) -> Mask:
    """
    Gives Track with parent values filled in by closest neighbor

    Args:
        - track
        - idx: locations of parent values to fill in

    See https://stackoverflow.com/questions/3662361/
    """
    if idx is None: idx = track < 0
    ind = distance_transform_edt(idx,
                                 return_distances=False,
                                 return_indices=True)
    # Cast to int to simplify indexing
    return track[tuple(ind)].astype(np.uint16)


def parents_from_track(track: Track) -> Dict[int, int]:
    """
    Returns dictionary of {daughter_id: parent_id}
    """
    # Parents are negative
    div_mask = (track * -1) > 0
    mask = track_to_mask(track, div_mask)

    # Ensure all keys and values will be int for indexing
    if track.dtype not in (np.int16, np.uint16):
        track = track.astype(np.int16)

    return dict(zip(mask[div_mask], track[div_mask] * -1))


def track_to_lineage(track: Track) -> np.ndarray:
    """
    Given a set of track images, reconstruct all the lineages
    """
    # Use cells to fill in info in lineage
    cells = np.unique(track[track > 0])

    # Find cells with parents
    parent_daughter_links = parents_from_track(track)
    parent_lookup = {c: 0 for c in cells}
    parent_lookup.update(parent_daughter_links)

    # Pre-allocate lineage
    # lineage[:, 1] = (label, first frame, last frame, parent)
    lineage = np.empty((len(cells), 4)).astype(np.uint16)
    for row, cell in enumerate(cells):
        frames = np.where(track == cell)[0]
        first, last = frames[0], frames[-1]

        lineage[row] = [cell, first, last, parent_lookup[cell]]

    return lineage


def lineage_to_track(mask: Mask,
                     lineage: np.ndarray
                     ) -> Track:
    """
    Each mask in each frame should have a pixel == -1 * parent

    TODO:
        - This won't work if area(region) <= ~6, depending on shape
    """
    out = mask.copy().astype(np.int16)
    for (lab, app, dis, par) in lineage:
        if par:
            # Get all pixels in the label
            lab_pxl = np.where(mask[app, ...] == lab)

            # Find the centroid and set to the parent value
            x = int(np.floor(np.sum(lab_pxl[0]) / len(lab_pxl[0])))
            y = int(np.floor(np.sum(lab_pxl[1]) / len(lab_pxl[1])))
            out[app, x, y] = -1 * par

    return out
