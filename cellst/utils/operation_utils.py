from typing import Dict, Generator, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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


def dilate_sitk(labels: Mask, radius: int) -> np.ndarray:
    """
    Direct copy from CellTK. Should dilate images

    TODO:
        - Update function (or at least understand the function)
    """
    slabels = sitk.GetImageFromArray(labels)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(radius)
    return sitk.GetArrayFromImage(gd.Execute(slabels))


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


def sliding_window_generator(arr: np.ndarray, overlap: int = 0) -> Generator:
    """
    NOTE: If memory is an issue here, can probably manually count the indices
          and make a generator that way, but it will probably be much slower.

    TODO:
        - Add low mem option (see above)
        - Add option to slide over different axis
    """
    # Shapes are all the same
    shape = (overlap + 1, *arr.shape[1:])
    # Create a generator, returns each cut of the array
    yield from [np.squeeze(s) for s in sliding_window_view(arr, shape)]


# TODO: Test including @numba.njit here
def shift_array(array: np.ndarray,
                shift: tuple,
                fill: float = np.nan,
                ) -> np.ndarray:
    """
    Shifts an array and fills in the values or crops to size

    See: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    result = np.empty_like(array)

    # Shift is along two axes
    y, x = int(shift[0]), int(shift[1])

    # TODO: This seems unesseccarily verbose
    if y == 0 and x == 0:
        result[:] = array
    elif y > 0 and x > 0:
        result[:y, :x] = fill
        result[y:, x:] = array[:-y, :-x]
    elif y > 0 and x < 0:
        result[:y, x:] = fill
        result[y:, :x] = array[:-y, -x:]
    elif y > 0 and x == 0:
        result[:y, :] = fill
        result[y:, :] = array[:-y, :]
    elif y < 0 and x > 0:
        result[y:, :x] = fill
        result[:y, x:] = array[-y:, :-x]
    elif y < 0 and x < 0:
        result[y:, x:] = fill
        result[:y, :x] = array[-y:, -x:]
    elif y < 0 and x == 0:
        result[y:, :] = fill
        result[:y, :] = array[-y:, :]
    elif y == 0 and x > 0:
        result[:, :x] = fill
        result[:, x:] = array[:, :-x]
    elif y == 0 and x < 0:
        result[y:, x:] = fill
        result[:, :x] = array[:, -x:]

    return result


def crop_array(array: np.ndarray,
               crop_vals: Tuple[int] = None,
               crop_area: float = 0.6
               ) -> np.ndarray:
    """
    Crops an image to the specified dimensions
    if crop_vals is None - use crop area to calc crop vals

    TODO:
        - There must be a much neater way to write this function
        - Incorporate crop_area
        - Possible numba.njit
    """
    if crop_vals is None:
        # TODO: calculate crop coordinates for area
        pass

    y, x = crop_vals

    if y == 0 and x == 0:
        return array
    elif y > 0 and x > 0:
        return array[..., y:, x:]
    elif y > 0 and x < 0:
        return array[..., y:, :x]
    elif y > 0 and x == 0:
        return array[..., y:, :]
    elif y < 0 and x > 0:
        return array[..., :y, x:]
    elif y < 0 and x < 0:
        return array[..., :y, :x]
    elif y < 0 and x == 0:
        return array[..., :y, :]
    elif y == 0 and x > 0:
        return array[..., x:]
    elif y == 0 and x < 0:
        return array[..., :x]


class RandomNameProperty():
    """
    This class is to be used with skimage.regionprops_table.
    Every extra property passed to regionprops_table must
    have a unique name, however, I want to use several as a
    placeholder, so that I can get the right size array, but fill
    in the values later. So, this assigns a random __name__.

    NOTE:
        - This does not guarantee a unique name, so getting IndexError
          in Extract is still possible.
    """
    def __init__(self, *args) -> None:
        rng = np.random.default_rng()
        # Make it extremely unlikely to get the same int
        self.__name__ = str(rng.integers(999999))

    @staticmethod
    def __call__(empty):
        return np.nan
