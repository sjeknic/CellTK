import numpy as np
from skimage.morphology import dilation, remove_small_holes
from skimage.measure import label
from scipy.ndimage import (binary_dilation, labeled_comprehension,
                           generic_filter)
import scipy.stats as stats
import SimpleITK as sitk

from cellst.utils._types import Image, Mask, Track, Arr


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


def track_to_lineage(track: Track) -> np.ndarray:
    """
    Given a set of track images, reconstruct all the lineages
    lineage.shape[1] is 4 for label, first frame, last frame, parent
    TODO:
        - Could be both faster and neater?
    """

    def _get_cell_info(f, x, y):
        f0, f1 = (np.min(f), np.max(f) + 1)
        x0, x1 = (np.min(x), np.max(x) + 1)
        y0, y1 = (np.min(y), np.max(y) + 1)

        test = track[f0:f1, x0:x1, y0:y1]
        try:
            par = -1 * test[test < 0][0]
        except IndexError:
            par = 0

        return f0, f1, par

    # Use cells to fill in info in lineage
    cells = np.unique(track[track > 0])
    # Pre-allocate lineage
    # lineage.shape[1] is 4 for label, first frame, last frame, parent
    lineage = np.empty((len(cells), 4)).astype(np.uint16)
    for row, cell in enumerate(cells):
        f, x, y = np.where(track == cell)

        lineage[row] = [cell, *_get_cell_info(f, x, y)]

    return lineage


def lineage_to_track(mask: Mask,
                     lineage: np.ndarray
                     ) -> Track:
    """
    Each mask in each frame should have a random(?) pixel
    set to the negative value of the parent cell.

    TODO:
        - This is less reliable with small regions
    """
    out = mask.copy().astype(np.int16)
    for (lab, app, dis, par) in lineage:
        if par:
            # Get all pixels in the label
            lab_pxl = np.where(mask[app, ...] == lab)

            # Find the centroid and set to the parent value
            # TODO: this won't work in all cases. trivial example if size==1
            x = int(np.floor(np.sum(lab_pxl[0]) / len(lab_pxl[0])))
            y = int(np.floor(np.sum(lab_pxl[1]) / len(lab_pxl[1])))
            out[app, x, y] = -1 * par

    return out
