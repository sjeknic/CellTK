from typing import Collection, Tuple

import numpy as np
import skimage.measure as meas
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, opening
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

from cellst.operation import Operation
from cellst.utils._types import Image, Mask
from cellst.utils.utils import image_helper
from cellst.utils.operation_utils import remove_small_holes_keep_labels, gray_fill_holes_celltk


class Segment(Operation):
    _input_type = (Image,)
    _output_type = Mask

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 input_masks: Collection[str] = [],
                 output: str = 'mask',
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        if isinstance(input_masks, str):
            self.input_masks = [input_masks]
        else:
            self.input_masks = input_masks

        self.output = output

    @staticmethod
    @image_helper
    def clean_labels(mask: Mask,
                     min_radius: float = 3,
                     max_radius: float = 15,
                     open_size: int = 3,
                     ) -> Mask:
        """
        Applies light cleaning. Removes small, large, and border-connected
        objectes. Applies opening.

        TODO:
            - Still getting some objects that are not contiguous.
        """
        min_area, max_area = np.pi * np.array((min_radius, max_radius)) ** 2
        out = np.empty(mask.shape).astype(np.uint16)
        for fr in range(mask.shape[0]):
            ma = mask[fr, ...]

            labels = remove_small_holes_keep_labels(ma, np.pi * min_radius ** 2)
            labels = clear_border(labels, buffer_size=2)
            pos = remove_small_objects(labels, min_area, connectivity=2)
            neg = remove_small_objects(labels, max_area, connectivity=2)
            pos[neg > 0] = 0
            labels = opening(pos, np.ones((open_size, open_size)))

            out[fr, ...] = labels

        return out

    # TODO: Should these methods actually be static? What's the benefit?
    @staticmethod
    @image_helper
    def constant_thres(image: Image,
                       thres: float = 1000,
                       negative: bool = False,
                       connectivity: int = 2
                       ) -> Mask:
        """
        TODO:
            - Do I have to explicitly set the output array to uint16?
        """
        if negative:
            test_arr = image <= thres
        else:
            test_arr = image >= thres

        # Need to iterate over frames, otherwise connections are
        # considered along the time axis as well.
        out = np.empty(image.shape).astype(np.uint16)
        for fr in range(image.shape[0]):
            out[fr, ...] = meas.label(test_arr[fr, ...],
                                      connectivity=connectivity)
        return out

    @staticmethod
    @image_helper
    def adaptive_thres(image: Image,
                       relative_thres: float = 0.1,
                       sigma: float = 50,
                       connectivity: int = 2
                       ) -> Mask:
        """
        Applies Gaussian blur to the image and selects pixels that
        are relative_thres larger than the blurred image.
        """
        out = np.empty(image.shape).astype(np.uint16)
        for fr in range(image.shape[0]):
            filt = gaussian_filter(image[fr, ...], sigma)
            filt = image[fr, ...] > filt * (1 + relative_thres)
            out[fr, ...] = meas.label(filt, connectivity=connectivity)

        return out

    @staticmethod
    @image_helper
    def otsu_thres(image: Image,
                   nbins: int = 256,
                   connectivity: int = 2
                   ) -> Mask:
        """
        Uses Otsu's method to determine the threshold. All pixels
        above the threshold are kept
        """
        out = np.empty(image.shape).astype(np.uint16)
        for fr in range(image.shape[0]):
            thres = threshold_otsu(image[fr, ...], nbins=nbins)
            out[fr, ...] = meas.label(image[fr, ...] > thres,
                                      connectivity=connectivity)

        return out

    @image_helper
    def unet_predict(self,
                     image: Image,
                     weight_path: str,
                     roi: (int, str) = 2,
                     batch: int = None,
                     classes: int = 3,
                     ) -> Image:
        """
        NOTE: If we had mulitple colors, then image would be 4D here. The Pipeline isn't
        set up for that now, so for now the channels is just assumed to be 1.

        roi - the prediction values are returned only for the roi
        batch - number of frames passed to model. None is all of them.
        classes - number of output categories from the model (has to match weights)
        """
        _roi_dict = {'background': 0, 'bg': 0, 'edge': 1,
                     'interior': 2, 'nuc': 2, 'cyto': 2}
        if isinstance(roi, str):
            try:
                roi = _roi_dict[roi]
            except KeyError:
                raise ValueError(f'Did not understand region of interest {roi}.')

        # Only import tensorflow and Keras if needed
        from cellst.utils.unet_model import UNetModel

        if not hasattr(self, 'model'):
            '''NOTE: If we had mulitple colors, then image would be 4D here.
            The Pipeline isn't set up for that now, so for now the channels
            is just assumed to be 1.'''
            channels = 1
            dims = (image.shape[1], image.shape[2], channels)

            self.model = UNetModel(dimensions=dims,
                                   weight_path=weight_path,
                                   model='unet')

        # Pre-allocate output memory
        # TODO: Incorporate the batch here.
        if batch is None:
            output = self.model.predict(image[:, :, :], roi=roi)
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a, roi=roi)
                                     for a in arrs], axis=0)

        # TODO: dtype can probably be downsampled from float32 before returning
        return output
