import warnings
from itertools import groupby
from typing import Union, Collection, Tuple

import numpy as np
import SimpleITK as sitk
import skimage.registration as regi
import skimage.restoration as rest
import skimage.filters as filt
import skimage.segmentation as segm
import skimage.util as util
import scipy.ndimage as ndi

from cellst.core.operation import BaseProcessor
from cellst.utils._types import Image, Mask, Track, Same
from cellst.utils.utils import ImageHelper
from cellst.utils.operation_utils import (sliding_window_generator,
                                          shift_array, crop_array, PadHelper,
                                          wavelet_background_estimate,
                                          wavelet_noise_estimate, cast_sitk)


class Processor(BaseProcessor):
    """
    TODO:
        - Add stand-alone crop function
        - Add optical-flow registration
        - Add N4 bias correction https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
        - Add flat fielding from reference
    """
    @ImageHelper(by_frame=False, as_tuple=True)
    def align_by_cross_correlation(self,
                                   image: Image = tuple([]),
                                   mask: Mask = tuple([]),
                                   track: Track = tuple([]),
                                   align_with: str = 'image',
                                   crop: bool = True
                                   ) -> Same:
        """
        Shifts and crops images based on regi.phase_cross_correlation.

        TODO:
            - Needs to confirm image shapes match before cropping,
              otherwise, on reruns image might be cropped multiple times
            - Make all inputs optional
        """
        sizes = [s.shape for s in image + mask + track]
        assert len(tuple(groupby(sizes))) == 1, 'Stacks must be same shape'

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

    @ImageHelper(by_frame=True, as_tuple=True)
    def tile_images(self,
                    image: Image = tuple([]),
                    mask: Mask = tuple([]),
                    track: Track = tuple([]),
                    layout: Tuple[int] = None,
                    border_value: Union[int, float] = 0.,
                    scaling: float = 1.
                    ) -> Image:
        """"""
        # TODO: Add scaling
        fil = sitk.TileImageFilter()
        fil.SetLayout(layout)
        fil.SetDefaultPixelValue(border_value)
        stacks = image + mask + track
        images = [cast_sitk(sitk.GetImageFromArray(s), 'sitkUInt16', True)
                  for s in stacks]
        out = fil.Execute(images)

        return sitk.GetArrayFromImage(out)

    @ImageHelper(by_frame=True)
    def gaussian_filter(self,
                        image: Image,
                        sigma: float = 2.5,
                        dtype: type = np.float32
                        ) -> Image:
        """
        Multidimensional Gaussian filter.

        TODO:
            - Test applying to a stack with sigma = (s1, s1, 0)
            - SimpleITK implementation should be faseter
        """
        return filt.gaussian(image, sigma, preserve_range=True,
                             output=np.empty(image.shape, dtype=dtype))

    @ImageHelper(by_frame=True)
    def binomial_blur(self,
                      image: Image,
                      iterations: int = 7
                      ) -> Image:
        """"""
        fil = sitk.BinomialBlurImageFilter()
        fil.SetRepetitions(iterations)

        img = fil.Execute(sitk.GetImageFromArray(image))
        return sitk.GetArrayFromImage(img)

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
    def uniform_filter(self,
                       image: Image,
                       size: int = 3,
                       mode: str = 'reflect',
                       cval: int = 0
                       ) -> Image:
        """"""
        return ndi.uniform_filter(image, size=size,
                                  mode=mode, cval=cval)

    @ImageHelper(by_frame=True)
    def rolling_ball_background_subtract(self,
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
    def curvature_anisotropic_diffusion(self,
                                        image: Image,
                                        iterations: int = 5,
                                        time_step: float = 0.125,
                                        conductance: float = 1.
                                        ) -> Image:
        """"""
        # Set up the filter
        fil = sitk.CurvatureAnisotropicDiffusionImageFilter()
        fil.SetNumberOfIterations(iterations)
        fil.SetTimeStep(time_step)
        fil.SetConductanceParameter(conductance)

        # Check the input image type and execute filter
        img = sitk.GetImageFromArray(image)
        out = cast_sitk(img, 'sitkFloat64', cast_up=True)
        out = fil.Execute(out)

        # Prevent float64 output to display in Fiji
        out = cast_sitk(out, 'sitkFloat32')
        out = sitk.GetArrayFromImage(out)

        return out

    @ImageHelper(by_frame=True)
    def inverse_gaussian_gradient(self,
                                  image: Image,
                                  alpha: float = 100.0,
                                  sigma: float = 5.0
                                  ) -> Image:
        """
        """
        return util.img_as_uint(
            segm.inverse_gaussian_gradient(image, alpha, sigma)
        )

    @ImageHelper(by_frame=True)
    def sobel_edge_detection(self,
                             image: Image,
                             orientation: str = 'both'
                             ) -> Image:
        """
        Applies Sobel filter

        orientation can be 'h', 'v', or 'both'

        TODO:
            - Could be run faster on whole stack
        """
        if orientation in ('h', 'horizontal'):
            sobel = filt.sobel_h(image)
        elif orientation in ('v', 'vertical'):
            sobel = filt.sobel_v(image)
        else:
            sobel = filt.sobel(image)

        return util.img_as_uint(sobel)

    @ImageHelper(by_frame=True)
    def sobel_edge_magnitude(self,
                             image: Image,
                             ) -> Image:
        """"""
        y = ndi.sobel(image, axis=1)
        x = ndi.sobel(image, axis=0)
        return np.hypot(x, y)

    @ImageHelper(by_frame=True)
    def roberts_edge_detection(self, image: Image) -> Image:
        """"""
        return filt.roberts(image)

    @ImageHelper(by_frame=True)
    def recurssive_gauss_gradient(self,
                                  image: Image,
                                  sigma: float = 1.,
                                  use_direction: bool = True
                                  ) -> Image:
        """"""
        # Set up the filter
        fil = sitk.GradientRecursiveGaussianImageFilter()
        fil.SetSigma(sigma)
        fil.SetUseImageDirection(use_direction)

        # Convert image and return
        # TODO: Type casting needed?
        im = sitk.GetImageFromArray(image)
        im = fil.Execute(im)
        im = sitk.GetArrayFromImage(im)

        # Get the total magnitude from both channels
        x, y = im[..., 0], im[..., 1]
        return np.hypot(x, y)

    @ImageHelper(by_frame=True)
    def recurssive_gauss_magnitude(self,
                                   image: Image,
                                   sigma: float = 1.,
                                   ) -> Image:
        """"""
        # Only constraint on type is to be Real
        fil = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
        fil.SetSigma(sigma)
        im = sitk.GetImageFromArray(image)
        im = fil.Execute(im)

        im = cast_sitk(im, 'sitkFloat32')
        return sitk.GetArrayFromImage(im)

    @ImageHelper(by_frame=True)
    def make_edge_potential_image(self,
                                  image: Image,
                                  method: str = 'sigmoid',
                                  alpha: float = None,
                                  beta: float = None,
                                  k1: float = None,
                                  k2: float = None
                                  ) -> Image:
        """
        method = sigmoid, exp, reciprocal

        K1 is the minimum value along the contour
        K2 is the avg value inside the countour
        K1 should be > K2

        alpha should be (K2 - K1) / 6
        beta should be (K1 + K2) / 2
        """
        img = sitk.GetImageFromArray(image)
        if method == 'sigmoid':
            if all([not a for a in (alpha, beta, k1, k2)]):
                # Need to estimate the values for the sigmoid params
                # Use Li Threshold to find ROI
                li = sitk.LiThresholdImageFilter()
                li.SetInsideValue(0)
                li.SetOutsideValue(1)
                _li = li.Execute(img)

                # Mask the Li region on the original image
                mask = sitk.MaskImageFilter()
                _ma = mask.Execute(img, _li)

                # Convert to array and use np to find values
                _arr = sitk.GetArrayFromImage(_ma)
                _arr = _arr[_arr > 0]
                k1 = np.percentile(_arr, 95 )
                k2 = np.mean(_arr)
                if k1 <= k2: warnings.warn('Sigmoid param estimation poor.')
                alpha = (k2 - k1) / 6.
                beta = (k1 + k2) / 2.
            elif alpha and beta:
                # Alpha and beta have preference over k1/k2
                pass
            elif k1 and k2:
                alpha = (k2 - k1) / 6
                beta = (k1 + k2) / 2
            else:
                raise ValueError('Must provide either alpha/beta or k1/k2.')

            # Set the Sigmoid filter
            fil = sitk.SigmoidImageFilter()
            fil.SetOutputMaximum(1.)
            fil.SetOutputMinimum(0.)
            fil.SetAlpha(alpha)
            fil.SetBeta(beta)
        elif method == 'exp':
            fil = sitk.ExpNegativeImageFilter()
        elif method == 'reciprocal':
            fil = sitk.BoundedReciprocalImageFilter()

        return sitk.GetArrayFromImage(fil.Execute(img))

    @ImageHelper(by_frame=True)
    def make_maurer_distance_map(self,
                                 image: Image,
                                 value_range: Collection[float] = None,
                                 inside_positive: bool = False,
                                 use_euclidian: bool = False,
                                 use_image_spacing: bool = False
                                 ) -> Image:
        """"""
        # Needs to be integer image in most cases
        img = sitk.GetImageFromArray(image)
        img = cast_sitk(img, 'sitkUInt16')

        sign = sitk.SignedMaurerDistanceMapImageFilter()
        sign.SetUseImageSpacing(False)
        return sitk.GetArrayFromImage(sign.Execute(img))

    @ImageHelper(by_frame=False)
    def histogram_matching(self,
                           image: Image,
                           bins: int = 1000,
                           match_pts: int = 100,
                           threshold: bool = False,
                           ref_frame: int = 0,
                           ) -> Image:
        """
        Histogram matching from CellTK
        """
        # Get frame that will set the histogram
        reference_frame = image[ref_frame]
        reference_frame = sitk.GetImageFromArray(reference_frame)

        # Get the histogram matching filter
        fil = sitk.HistogramMatchingImageFilter()
        fil.SetNumberOfHistogramLevels(bins)
        fil.SetNumberOfMatchPoints(match_pts)
        fil.SetThresholdAtMeanIntensity(threshold)

        # Make output array
        out = np.empty_like(image)

        # Then iterate through all images
        for idx, frame in enumerate(image):
            # Apply the filter and save
            im = sitk.GetImageFromArray(frame)
            filimg = fil.Execute(im, reference_frame)
            out[idx, ...] = sitk.GetArrayFromImage(filimg)

        return out

    @ImageHelper(by_frame=False)
    def wavelet_background_subtract(self,
                                    image: Image,
                                    wavelet: str = 'db4',
                                    mode: str = 'symmetric',
                                    level: int = None,
                                    blur: bool = False,
                                    ) -> Image:
        """
        """
        # Pad image to even before starting
        padder = PadHelper(target='even', axis=[1, 2], mode='edge')
        image_pad = padder.pad(image)

        # Pass frames of the padded image
        out = np.zeros(image_pad.shape, dtype=image.dtype)
        for fr, im in enumerate(image_pad):
            bg = wavelet_background_estimate(im, wavelet, mode,
                                             level, blur)
            bg = np.asarray(bg, dtype=out.dtype)

            # Remove background and ensure non-negative
            out[fr, ...] = im - bg
            out[fr, ...][out[fr, ...] < 0] = 0

        # Undo padding and reset dtype before return
        return padder.undo_pad(out.astype(image.dtype))

    @ImageHelper(by_frame=False)
    def wavelet_noise_subtract(self,
                               image: Image,
                               noise_level: int = 1,
                               thres: int = 2,
                               wavelet: str = 'db1',
                               mode: str = 'smooth',
                               level: int = None,
                               ) -> Image:
        """
        """
        # Pad image to even before starting
        padder = PadHelper(target='even', axis=[1, 2], mode='edge')
        image_pad = padder.pad(image)

        # Pass frames of the padded image
        out = np.zeros(image_pad.shape, dtype=image.dtype)
        for fr, im in enumerate(image_pad):
            ns = wavelet_noise_estimate(im, noise_level, wavelet,
                                        mode, level, thres)
            ns = np.asarray(ns, dtype=out.dtype)

            # Remove background and ensure non-negative
            out[fr, ...] = im - ns
            out[fr, ...][out[fr, ...] < 0] = 0

        # Undo padding and reset dtype before return
        return padder.undo_pad(out.astype(image.dtype))

    @ImageHelper(by_frame=False)
    def unet_predict(self,
                     image: Image,
                     weight_path: str,
                     roi: Union[int, str] = 2,
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
        from cellst.utils.unet_model import FluorUNetModel

        if not hasattr(self, 'model'):
            '''NOTE: If we had mulitple colors, then image would be 4D here.
            The Pipeline isn't set up for that now, so for now the channels
            is just assumed to be 1.'''
            channels = 1
            dims = (image.shape[1], image.shape[2], channels)

            self.model = FluorUNetModel(dimensions=dims,
                                        weight_path=weight_path)

        # Pre-allocate output memory
        if batch is None:
            output = self.model.predict(image[:, :, :], roi=roi)
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a, roi=roi)
                                     for a in arrs], axis=0)

        # TODO: dtype can probably be downsampled from float32 before returning
        return output
