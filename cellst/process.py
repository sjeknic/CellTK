import numpy as np
import SimpleITK as sitk
import skimage.registration as regi
import skimage.restoration as rest
import skimage.filters as filt
import skimage.segmentation as segm
import scipy.ndimage as ndi

from cellst.core.operation import BaseProcessor
from cellst.utils._types import Image, Mask, Track, Same
from cellst.utils.utils import ImageHelper
from cellst.utils.operation_utils import (sliding_window_generator,
                                          shift_array, crop_array)


class Processor(BaseProcessor):
    """
    TODO:
        - Add stand-alone crop function
        - Add optical-flow registration
        - Add faster bg subtract (wavelet hazen)
        - Add sobel filter
    """
    @ImageHelper(by_frame=False, as_tuple=True)
    def align_by_cross_correlation(self,
                                   image: Image,
                                   mask: Mask = tuple([]),
                                   track: Track = tuple([]),
                                   align_with: str = 'image',
                                   crop: bool = True
                                   ) -> Same:
        """
        Shifts and crops images based on regi.phase_cross_correlation.
        """
        # Image that aligning will be based on - first img in align_with
        to_align = locals()[align_with][0]

        # Get frame generator
        frame_generator = sliding_window_generator(to_align, overlap=1)

        # Calculate shifts using phase cross correlation
        shifts = []
        for idx, frames in enumerate(frame_generator):
            # frame_generator yields array of shape (overlap, y, x)
            shifts.append(regi.phase_cross_correlation(frames[0, ...],
                                                       frames[1, ...])[0])

        # Get all shifts relative to the first image (cumulative)
        shifts = np.vstack(shifts)
        cumulative = np.cumsum(shifts, axis=0)

        # Make arrays for each output
        flat_inputs = image + mask + track
        flat_outputs = [np.empty_like(arr) for arr in flat_inputs]

        # Copy first frames and store the output
        for fi, fo in zip(flat_inputs, flat_outputs):
            fo[0, ...] = fi[0, ...]
            # First frame is handled, iterate over the rest
            for idx, (cumul, frame) in enumerate(zip(cumulative, fi[1:])):
                fo[idx + 1, ...] = shift_array(frame, cumul, fill=0)

        # Crop the whole stack together if needed
        if crop:
            # Crop is the largest shift in each axis
            crop_idx = (np.argmax(np.abs(cumulative[:, 0])),
                        np.argmax(np.abs(cumulative[:, 1])))
            crop_vals = (int(cumulative[crop_idx[0], 0]),
                         int(cumulative[crop_idx[1], 1]))
            flat_outputs = [crop_array(fo, crop_vals) for fo in flat_outputs]

        return flat_outputs

    @ImageHelper(by_frame=True)
    def gaussian_filter(self,
                        image: Image,
                        sigma: float = 2.5
                        ) -> Image:
        """
        Multidimensional Gaussian filter

        TODO:
            - Test applying to a stack with sigma = (s1, s1, 0)
            - SimpleITK implementation should be faseter
        """
        return filt.gaussian(image, sigma)

    @ImageHelper(by_frame=True)
    def gaussian_laplace_filter(self,
                                image: Image,
                                sigma: float = 2.5,
                                ) -> Image:
        """
        Multidimensional Laplace filter using Gaussian second derivatives.

        TODO:
            - Test applying to a stack with sigma = (s1, s1, 0)
            - SimpleITK implementation should be faster
        """
        return ndi.gaussian_laplace(image, sigma)

    @ImageHelper(by_frame=True)
    def rolling_ball_background_subtraction(self,
                                            image: Image,
                                            radius: float = 100,
                                            kernel: np.ndarray = None,
                                            nansafe: bool = False
                                            ) -> Image:
        """
        Estimate background intensity by rolling/translating a kernel.

        TODO:
            - Check CellTK, this function did a lot more for some reason
        """
        bg = rest.rolling_ball(image, radius=radius,
                               kernel=kernel, nansafe=nansafe)
        return image - bg

    @ImageHelper(by_frame=True)
    def inverse_gaussian_gradient(self,
                                  image: Image,
                                  alpha: float = 100.0,
                                  sigma: float = 5.0
                                  ) -> Image:
        """
        """
        return segm.inverse_gaussian_gradient(image, alpha, sigma)

    @ImageHelper(by_frame=True, overlap=1)
    def histogram_matching(self,
                           image: Image,
                           bins: int = 500,
                           match_pts: int = 2,
                           threshold: bool = False,
                           ) -> Image:
        """
        Histogram matching from CellTK
        """
        # Get frames as sITK images
        frame0 = sitk.GetImageFromArray(image[0, ...])
        frame1 = sitk.GetImageFromArray(image[1, ...])

        # Get the histogram matching filter
        fil = sitk.HistogramMatchingImageFilter()
        fil.SetNumberOfHistogramLevels(bins)
        fil.SetNumberOfMatchPoints(match_pts)
        fil.SetThresholdAtMeanIntensity(threshold)

        # Apply the filter and return
        filimg = fil.Execute(frame0, frame1)
        return sitk.GetArrayFromImage(filimg)

    @ImageHelper(by_frame=False)
    def unet_predict(self,
                     image: Image,
                     weight_path: str,
                     roi: (int, str) = 2,
                     batch: int = None,
                     classes: int = 3,
                     ) -> Image:
        """
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
