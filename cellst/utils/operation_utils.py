from typing import Dict, Generator, Tuple, List, Iterable, Union

import numpy as np
import pywt
import numpy.lib.stride_tricks as stricks
import skimage.morphology as morph
import skimage.measure as meas
import skimage.segmentation as segm
import scipy.ndimage as ndi
import scipy.optimize as opti
import mahotas.segmentation as mahotas_seg
import SimpleITK as sitk

from cellst.utils._types import Mask, Track

# TODO: Add label by parent function

def gray_fill_holes(labels: np.ndarray) -> np.ndarray:
    """
    Faster (but hopefully identical) to the CellTK version
    """
    fil = sitk.GrayscaleFillholeImageFilter()
    filled = sitk.GetArrayFromImage(
        fil.Execute(sitk.GetImageFromArray(labels))
    )
    idx = np.where(filled != labels, True, False)
    idx = ndi.distance_transform_edt(idx,
                                     return_distances=False,
                                     return_indices=True)

    return labels[tuple(idx)]


def dilate_sitk(labels: np.ndarray, radius: int) -> np.ndarray:
    """
    Direct copy from CellTK. Should dilate images
    """
    slabels = sitk.GetImageFromArray(labels)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(radius)
    return sitk.GetArrayFromImage(gd.Execute(slabels))


def _np_types() -> dict:
    _np_dtypes = {'integer': (np.uint, np.uint8, np.uint16,
                              np.uint32, np.uint64),
                  'sinteger': (int, np.int, np.int8, np.int16,
                               np.int32, np.int64),
                  'float': (np.single, np.float32, np.double,
                            np.float64, np.float128),
                  'complex': (np.csingle, np.complex64, np.cdouble,
                              np.complex128, np.cfloat)}
    return _np_dtypes


def _sitk_types(test: Union[str, "sitk.BasicPixelID"] = None) -> dict:
    _sitk_types = {'integer': (sitk.sitkUInt8, sitk.sitkUInt16,
                               sitk.sitkUInt32, sitk.sitkUInt64),
                   'sinteger': (sitk.sitkInt8, sitk.sitkInt16,
                                sitk.sitkInt32, sitk.sitkInt64),
                   'float': (sitk.sitkFloat32, sitk.sitkFloat64),
                   'complex': (sitk.sitkComplexFloat32,
                               sitk.sitkComplexFloat64)}

    if test is None:
        return _sitk_types
    else:
        if isinstance(test, str):
            test = getattr(sitk, test)

        return [k for k, v in _sitk_types.items()
                if test in v][0]


def _sitk_enum_to_string(test: int) -> str:
    # https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html
    # #ae40bd64640f4014fba1a8a872ab4df98
    _inorder_list = ['sitkInt8', 'sitkUInt8', 'sitkInt16',
                     'sitkUInt16', 'sitkInt32', 'sitkUInt32', 'sitkInt64',
                     'sitkUInt64', 'sitkFloat32', 'sitkFloat64',
                     'sitkComplexFloat32', 'sitkComplexFloat64']
    _pixel_values = {i: p for i, p in enumerate(_inorder_list)}

    return _pixel_values[test]


def _casting_up(inp: Union[str, int], out: Union[str, int]) -> bool:
    # Returns trailing digits
    def _digit(test) -> int:
        i = 0
        while True:
            i -= 1
            if test[i].isdigit():
                pass
            else:
                i += 1  # stopped on non-digit
                break

        return int(test[i:])

    # Get everything in strings
    if isinstance(inp, int):
        inp = _sitk_enum_to_string(inp)
    if isinstance(inp, int):
        out = _sitk_enum_to_string(out)

    # Get groups
    igrp = _sitk_types(inp)
    ogrp = _sitk_types(out)

    # By default
    cast_up = False

    if igrp in ('integer', 'sinteger') and ogrp in ('float', 'complex'):
        cast_up = True
    elif igrp in ('float') and ogrp in ('complex'):
        cast_up = True
    elif igrp in ('integer', 'sinteger') and ogrp in ('integer', 'sinteger'):
        cast_up = _digit(inp) < _digit(out)
    elif igrp in ('float') and ogrp in ('float'):
        cast_up = _digit(inp) < _digit(out)
    elif igrp in ('complex') and ogrp in ('complex'):
        cast_up = _digit(inp) < _digit(out)

    return cast_up


def get_image_pixel_type(image: Union[np.ndarray, sitk.Image]) -> str:
    """"""
    _np_dtypes = _np_types()
    _sitk_dtypes = _sitk_types()

    try:
        if isinstance(image, np.ndarray):
            pxl = image.dtype
            key = [k for k, v in _np_dtypes.items()
                   if pxl in v][0]
        elif isinstance(image, sitk.Image):
            pxl = sitk.GetPixelIDType()
            key = [k for k, v in _sitk_dtypes.items()
                   if pxl in v][0]
        else:
            raise IndexError
    except IndexError:
        raise TypeError('Did not understand type of '
                        f'input image {type(image)}')

    return key


def cast_sitk(image: sitk.Image,
              req_type: str,
              cast_up: bool = False
              ) -> sitk.Image:
    """"""
    # Get the relevant types
    # This returns an integer of the required type
    input_type = _sitk_enum_to_string(image.GetPixelIDValue())
    assert hasattr(sitk, req_type)

    # Check if casting up for early exit
    if not cast_up:
        up = _casting_up(input_type, req_type)
        # Requested type is greater than input type
        if up:
            return image

    # Cast and return
    if input_type != req_type:
        cast = sitk.CastImageFilter()
        cast.SetOutputPixelType(getattr(sitk, req_type))
        image = cast.Execute(image)

    return image


def _close_border_holes(array: np.ndarray,
                        max_length: int = 45,
                        in_place: bool = True
                        ) -> np.ndarray:
    """"""
    if not in_place: array = array.copy()

    axes = (array[0, :], array[-1, :],  # top, bottom
            array[:, 0], array[:, -1])  # left, right
    for ax in axes:
        # Find holes by comparing to all indices
        nonzero = np.where(ax)[0]
        holes = np.setdiff1d(np.arange(len(ax)), nonzero)

        if len(holes):
            # Find holes split them up to be unique
            diffs = np.ediff1d(holes, to_begin=1)
            nonzero_diffs = np.ediff1d(holes, to_begin=1)


            hole_idxs = np.split(holes, np.where(diffs > 1)[0])

            # Fill them in
            for h in hole_idxs:
                #TODO: Add minlength of filled in criteria
                if len(h) <= max_length:
                    ax[h] = 1

    return array


def sitk_binary_fill_holes(labels: np.ndarray,
                           fill_border: bool = True,
                           iterations: Union[int, bool] = False,
                           kernel_radius: int = 4,
                           max_length: int = 45,
                           in_place: bool = True,
                           **kwargs
                           ) -> np.ndarray:
    """
    TODO:
        - Add lots of options
        - Add VoteIterativeHoleFilling
        - Add closing/opening

        - Should iterations be first or last?
    """
    if iterations:
        fil = sitk.VotingBinaryIterativeHoleFillingImageFilter()
        fil.SetMaximumNumberOfIterations(iterations)
        fil.SetRadius(kernel_radius)
    else:
        fil = sitk.BinaryFillholeImageFilter()

    # kwargs are used to set values on the filters
    for k, v in kwargs.items():
        getattr(fil, k)(v)

    # Fill the holes first
    if isinstance(labels, np.ndarray):
        _labels = sitk.GetImageFromArray(labels)
    elif isinstance(labels, sitk.Image):
        _labels = labels

    labels = fil.Execute(_labels)

    if fill_border:
        # Close any border holes
        labels = sitk.GetArrayFromImage(labels)
        labels = _close_border_holes(labels, max_length, in_place)

        labels = np.pad(labels, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        # Re-fill so that those border holes are filled
        fil = sitk.VotingBinaryHoleFillingImageFilter()
        fil.SetRadius(kernel_radius)
        _labels = sitk.GetImageFromArray(labels)
        labels = fil.Execute(_labels)

        labels = labels[1:-1, 1:-1]

    return sitk.GetArrayFromImage(labels)


def ndi_binary_fill_holes(labels: np.ndarray,
                          fill_border: bool = True,
                          kernel_radius: int = 2,
                          max_length: int = 45,
                          in_place: bool = False,
                          ) -> np.ndarray:
    """"""
    labels = ndi.binary_fill_holes(labels, get_binary_footprint(kernel_radius))

    if fill_border:
        labels = _close_border_holes(labels, max_length, in_place)
        labels = np.pad(labels, ((1, 1), (1, 1)),
                        mode='constant', constant_values=0)
        labels = ndi.binary_fill_holes(labels,
                                       get_binary_footprint(kernel_radius)
                                       )
        labels = morph.binary_opening(labels)
        labels = labels[1:-1, 1:-1]

    return labels


def mask_to_seeds(mask: np.ndarray,
                  method: str = 'sitk',
                  output: str = 'mask',
                  binary: bool = True) -> Union[np.ndarray, list]:
    """Find centroid of all objects and return, either as list of points or labeled mask
    If binary, all seeds are 1, otherwise, preserve labels

    Currently, none of the options d anything
    """
    if method == 'sitk':
        img = sitk.GetImageFromArray(mask)
        img = cast_sitk(img, 'sitkUInt16')
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(img)
        centroids = [stats.GetCentroid(i) for i in stats.GetLabels()]
        pixels = [stats.GetNumberOfPixels(i) for i in stats.GetLabels()]
        perim = [stats.GetPerimeter(i) for i in stats.GetLabels()]

    if output == 'mask':
        out = np.zeros_like(mask)
        pt0, pt1 = zip(*centroids)
        pt0 = np.array([int(round(p)) for p in pt0])
        pt1 = np.array([int(round(p)) for p in pt1])

        out[pt1, pt0] = 1

    return out


def track_to_mask(track: Track, idx: np.ndarray = None) -> Mask:
    """
    Gives Track with parent values filled in by closest neighbor

    Args:
        - track
        - idx: locations of parent values to fill in

    See https://stackoverflow.com/questions/3662361/
    """
    if idx is None: idx = track < 0
    ind = ndi.distance_transform_edt(idx,
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

    See CellTrackingChallenge and BayesianTracker.ldep for possible
    formats of the lineage array.

    TODO:
        - This won't work if area(region) <= ~6, depending on shape
        - Also might not work for discontinuous regions
    """
    out = mask.copy().astype(np.int16)
    for (lab, app, dis, par) in lineage:
        if par and par != lab:  # Had to change to accomodate bayes track
            # Get all pixels in the label
            lab_pxl = np.where(mask[app, ...] == lab)

            # Find the centroid and set to the parent value
            x = int(np.floor(np.sum(lab_pxl[0]) / len(lab_pxl[0])))
            y = int(np.floor(np.sum(lab_pxl[1]) / len(lab_pxl[1])))
            out[app, x, y] = -1 * par

    return out


def sliding_window_generator(arr: np.ndarray, overlap: int = 0) -> Generator:
    """
    overlap: int(amount of frames to overlap between passing)
    e.g. overlap = 1: [0, 1], [1, 2], [2, 3], [3, 4]
         overlap = 2: [0, 1, 2], [1, 2, 3], [2, 3, 4]

    NOTE: Overlaps get passed as a stack, not as separate args.
          i.e. if overlap = 1, image.shape = (2, h, w)
    NOTE: If memory is an issue here, can probably manually count the indices
          and make a generator that way, but it will be much slower.

    TODO:
        - Add low mem option (see above)
        - Add option to slide over different axis, by default uses 0
    """
    if overlap:
        # Shapes are all the same
        shape = (overlap + 1, *arr.shape[1:])
        # Create a generator, returns each cut of the array
        yield from [np.squeeze(s)
                    for s in stricks.sliding_window_view(arr, shape)]
    else:
        yield from arr


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


def voronoi_boundaries(seed: np.ndarray,
                       thin: bool = False,
                       thick: bool = False,) -> np.ndarray:
    """
    Calculate voronoi boundaries, and return as mask to set pixels to 0.
    """
    bound = segm.find_boundaries(mahotas_seg.gvoronoi(seed))

    if thin:
        bound = morph.thin(bound)
    if thick:
        bound = morph.binary_dilation(bound.astype(bool),
                                      footprint=np.ones((3, 3)))
    return bound


def skimage_level_set(shape: Tuple[int],
                      levelset: str = 'checkerboard',
                      size: (float, int) = None,
                      center: Tuple[int] = None,
                      ) -> np.ndarray:
    """
    Wrapper for levelset functions in skimage.segmentation

    size refers to square_size for checkerboard or radius for disk
    """
    if levelset == 'checkerboard':
        size = int(size) if size else 5  # default for skimage
        out = segm.checkerboard_level_set(shape, size)
    elif levelset == 'disk':
        out = segm.disk_level_set(shape, center, size)
    else:
        raise ValueError(f'Could not find level_set function for {levelset}')

    return out


def get_binary_footprint(rank: int = 2, connectivity: int = 1) -> np.ndarray:
    """Wrapper for ndi.generate_binary_structure"""
    assert connectivity <= rank
    return ndi.generate_binary_structure(rank, connectivity)


def match_labels_linear(source: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """
    Should transfer labels from source to dest based on area overlap

    TODO:
        - Should there be a threshold of the overlapping amount?
        - Should overlap be relative to source or dest?
        - Handle overflow amounts
    """
    # Get unique labels and remove 0
    source_labels = np.unique(source)[1:]
    dest_labels = np.unique(dest)[1:]
    dest_idx = {d: n for n, d in enumerate(dest_labels)}

    # Calculate matrix of overlaps
    cost_matrix = np.zeros((len(source_labels), len(dest_labels)))
    for x, slab in enumerate(source_labels):
        # Get values in dest that overlap with slab
        _dest = dest[source == slab]
        _area = np.count_nonzero(_dest)
        labels, overlaps = np.unique(_dest[_dest > 0], return_counts=True)

        # Need to remove 0 again
        for l, o in zip(labels, overlaps):
            cost_matrix[x, dest_idx[l]] = -o / _area

    # These are the indices of the lowest cost assignment
    s_idx, d_idx = opti.linear_sum_assignment(cost_matrix)

    # Check if all dest labels were labeled
    # TODO: Should add option to check source labels
    if len(d_idx) < len(dest_labels):
        # Get the indices of the unlabled and add to the original.
        unlabeled = set(range(len(dest_labels))).difference(d_idx)
        unlabeled = np.fromiter(unlabeled, int)
        new_labels = np.arange(1, len(unlabeled) + 1) + np.max(source_labels)

        # Update the original arrays
        source_labels = np.concatenate([source_labels, new_labels])
        d_idx = np.concatenate([d_idx, unlabeled])
        s_idx = np.concatenate([s_idx,
                                np.arange(len(s_idx), len(source_labels))])

    # Assign the values in a new output matrix
    out = np.zeros_like(dest)
    for s, d in zip(source_labels[s_idx], dest_labels[d_idx]):
        out[dest == d] = s

    return out


def wavelet_background_estimate(image: np.ndarray,
                                wavelet: str = 'db4',
                                mode: str = 'smooth',
                                level: int = None,
                                blur: bool = False,
                                axes: Tuple[int] = (-2, -1)
                                ) -> np.ndarray:
    """"""
    # Get approximation and detail coeffecients
    coeffs = pywt.wavedec2(image, wavelet, mode=mode,
                           level=level, axes=axes)

    # Set detail coefficients to 0
    for idx, coeff in enumerate(coeffs):
        if idx:  # skip first coefficients
            coeffs[idx] = tuple([np.zeros_like(c) for c in coeff])

    # Reconstruct and blur if needed
    bg = pywt.waverec2(coeffs, wavelet, mode)
    if blur:
        # If level is undefined, estimate here
        if not level:
            level = np.min([pywt.dwt_max_level(image.shape[a], wavelet)
                            for a in axes])
        sig = 2 ** level
        bg = ndi.gaussian_filter(bg, sig)

    return bg


def wavelet_noise_estimate(image: np.ndarray,
                           noise_level: int = 1,
                           wavelet: str = 'db1',
                           mode: str = 'smooth',
                           level: int = None,
                           thres: int = 2,
                           axes: Tuple[int] = (-2, -1),
                           ) -> np.ndarray:
    """"""
    # Get approximation and detail coeffecients
    coeffs = pywt.wavedec2(image, wavelet, mode=mode,
                           level=level, axes=axes)

    # Set detail coefficients to 0
    for idx, coeff in enumerate(coeffs[:-noise_level]):
        if idx:  # skip first coefficients
            coeffs[idx] = tuple([np.zeros_like(c) for c in coeff])
        else:
            coeffs[idx] = np.ones_like(coeff)

    # Reconstruct and blur if needed
    noise = pywt.waverec2(coeffs, wavelet, mode)

    # Apply threshold compared to standard deviation of noise
    thres_val = np.mean(noise) + thres * np.std(noise)
    noise[noise > thres_val] = thres_val

    return noise


class PadHelper():
    """
    TODO:
        - Add more complex padding options (e.g. pad both side, etc)
        - Move this function to utils or something
    """
    def __init__(self,
                 target: (str, int),
                 axis: (int, List[int]) = None,
                 mode: str = 'constant',
                 **kwargs
                 ) -> None:
        # Target can be 'even', 'odd', or a specific shape
        self.target = target
        self.mode = mode
        self.kwargs = kwargs

        # If axis is None, applies to all, otherwise, just some
        if not isinstance(axis, Iterable):
            self.axis = tuple([axis])
        else:
            self.axis = axis

        # No pads yet
        self.pads = []

    def pad(self, arr: np.ndarray) -> np.ndarray:
        """"""
        # Pad always rewrites self.pads
        self.pads = self._calculate_pads(arr.shape)

        return np.pad(arr, self.pads, self.mode, **self.kwargs)

    def undo_pad(self, arr: np.ndarray) -> np.ndarray:
        """"""
        if not self.pads:
            raise ValueError('Pad values not found.')

        pads_r = self._reverse_pads(self.pads)
        # Turn pads_r into slices for indexing
        slices = [slice(None)] * len(pads_r)
        for n, (st, en) in enumerate(pads_r):
            if not st and not en:
                continue
            else:
                slices[n] = slice(st, en)

        return arr[tuple(slices)]

    def _calculate_pads(self, shape: Tuple[int]) -> Tuple[int]:
        """"""
        if not self.axis:
            # If no axis is specified, pad all of them
            self.axis = range(len(shape))

        pads = [(0, 0)] * len(shape)
        for ax in self.axis:
            sh = shape[ax]
            if self.target == 'even':
                pads[ax] = (0, int(sh % 2))
            elif self.target == 'odd':
                pads[ax] = (0, int(not sh % 2))
            else:
                # self.target is a number
                pads[ax] = (0, int(self.target - sh))

        return pads

    def _reverse_pads(self, pads: Tuple[int]) -> Tuple[int]:
        """"""
        pads_r = [(0, 0)] * len(pads)
        for n, pad in enumerate(pads):
            pads_r[n] = tuple([int(-1 * p) for p in pad])

        return pads_r


def _remove_small_holes_keep_labels(image: np.ndarray,
                                    size: float
                                    ) -> np.ndarray:
    """
    Wrapper for skimage.morphology.remove_small_holes
    to keep the same labels on the images.

    Probably is not a good way to do this, but kept for
    now for debugging purposes.

    TODO:
        - Confirm correct selem to use or make option
    """
    dilated = morph.dilation(image, selem=np.ones((3, 3)))
    fill = morph.remove_small_holes(image, area_threshold=size,
                                    connectivity=2, in_place=False)
    return np.where(fill > 0, dilated, 0)


def _gray_fill_holes_celltk(labels):
    """
    Direct copy from CellTK. Should not be used in Pipeline.
    Kept for now for debugging purposes.
    """
    fil = sitk.GrayscaleFillholeImageFilter()
    filled = sitk.GetArrayFromImage(fil.Execute(sitk.GetImageFromArray(labels)))
    holes = meas.label(filled != labels)
    for idx in np.unique(holes):
        if idx == 0:
            continue
        hole = holes == idx
        surrounding_values = labels[ndi.binary_dilation(hole) & ~hole]
        uniq = np.unique(surrounding_values)
        if len(uniq) == 1:
            labels[hole > 0] = uniq[0]
    return labels
