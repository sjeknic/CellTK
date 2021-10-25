import numpy as np
from skimage.morphology import dilation, remove_small_holes
from scipy.ndimage import binary_dilation
from skimage.measure import label

import SimpleITK as sitk


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
    fill = remove_small_holes(image, area_threshold=10,
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
