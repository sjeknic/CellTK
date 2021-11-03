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
        - Feels like this is clunky. Could be both faster and neater
    """
    def _get_daughter(idx):
        # index 0 is frames
        x_idx = idx[1]
        y_idx = idx[2]

        # Make indices for cross positions
        left_idx = tuple([idx[0], x_idx - 1, y_idx])
        right_idx = tuple([idx[0], x_idx + 1, y_idx])
        up_idx = tuple([idx[0], x_idx, y_idx + 1])
        down_idx = tuple([idx[0], x_idx, y_idx - 1])
        cross_idx = [left_idx, right_idx, up_idx, down_idx]

        # Get possible values
        poss = np.array([track[c] for c in cross_idx])
        poss = poss[poss != 0]
        # TODO: This will fail if the object is very small.
        #       probably default to using mode or np unique
        assert all([p == poss[0] for p in poss]), 'Ambiguous parent id'

        return poss[0]

    # Pre-allocate lineage
    cells = np.unique(track[track > 0])

    # Use cells to fill in info in lineage
    # lineage.shape[1] is 4 for label, first frame, last frame, parent
    lineage = np.empty((len(cells), 4)).astype(np.int16)
    for row, cell in enumerate(cells):
        frames = np.where(track == cell)[0]
        frst, last = (np.min(frames), np.max(frames))

        lineage[row, 0] = cell
        lineage[row, 1] = frst
        lineage[row, 2] = last
        lineage[row, 3] = 0  # default for cells with no parent

    # Each negative pixel should uniquely id a parent
    negatives = np.argwhere(track < 0)
    for neg in negatives:
        par = track[tuple(neg)]
        assert par < 0, 'Incorrect parent index'
        dau = _get_daughter(neg)
        row = np.where(lineage == dau)[0]
        assert len(row) == 1, 'Ambiguous daughter id'

        lineage[row, 3] = -1 * par

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
