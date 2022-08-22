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
import sklearn.preprocessing as preproc

from celltk.core.operation import BaseProcess
from celltk.utils._types import Image, Mask, Stack, Optional
from celltk.utils.utils import ImageHelper
from celltk.utils.operation_utils import (sliding_window_generator,
                                          shift_array, crop_array, PadHelper,
                                          wavelet_background_estimate,
                                          wavelet_noise_estimate, cast_sitk,
                                          get_image_pixel_type)


class Process(BaseProcess):
    """
    TODO:
        - Add stand-alone crop function
        - Add optical-flow registration
        - Add flat fielding from reference
        - Add rescaling intensity stand alone
    """
    @ImageHelper(by_frame=False, as_tuple=True)
    def align_by_cross_correlation(self,
                                   image: Image = tuple([]),
                                   mask: Mask = tuple([]),
                                   align_with: str = 'image',
                                   crop: bool = True,
                                   normalization: str = 'phase'
                                   ) -> Stack:
        """Uses phase cross-correlation to shift the images to align them.
        Optionally can crop the images to align. Align with can be used
        to specify which of the inputs to use. Uses the first stack in the
        given list.

        :param image: List of image stacks to be aligned.
        :param mask: List of mask stacks to be aligned.
        :param align_with: Can be one of 'image', 'mask', or 'track'. Defines
            which of the input stacks should be used for alignment.
        :param crop: If True, the aligned stacks are cropped based on the
            largest frame to frame shifts.
        :param normalization:

        :return: Aligned input stack.

        :raises AssertionError: If input stacks have different shapes.

        TODO:
            - Needs to confirm image shapes match before cropping,
              otherwise, on reruns image might be cropped multiple times
            - Make all inputs optional
        """
        sizes = [s.shape for s in image + mask]
        assert len(tuple(groupby(sizes))) == 1, 'Stacks must be same shape'

        # Image that aligning will be based on - first img in align_with
        to_align = locals()[align_with][0]

        # Get frame generator
        frame_generator = sliding_window_generator(to_align, overlap=1)

        # Calculate shifts using phase cross correlation
        shifts = []
        for idx, frames in enumerate(frame_generator):
            # frame_generator yields array of shape (overlap, y, x)
            shifts.append(
                regi.phase_cross_correlation(frames[0, ...], frames[1, ...],
                                             normalization=normalization)[0]
            )

        # Get all shifts relative to the first image (cumulative)
        shifts = np.vstack(shifts)
        cumulative = np.cumsum(shifts, axis=0)

        # Make arrays for each output
        flat_inputs = image + mask
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

    @ImageHelper(by_frame=False, as_tuple=True)
    def crop_to_area(self,
                     image: Image[Optional] = tuple([]),
                     mask: Mask[Optional] = tuple([]),
                     crop_factor: float = 0.6
                     ) -> Stack:
        """
        """
        flat_inputs = image + mask
        sizes = [s.shape for s in flat_inputs]
        assert len(tuple(groupby(sizes))) == 1, 'Stacks must be same shape'
        _, x, y = sizes[0]

        axis_factor = crop_factor ** 0.5
        x_amount = np.floor((axis_factor * x) / 2)
        y_amount = np.floor((axis_factor * y) / 2)
        crop_vals = (int(x_amount), int(y_amount))

        out = [crop_array(f, crop_vals) for f in flat_inputs]
        out = [crop_array(f, (-1 * crop_vals[0], -1 * crop_vals[1]))
               for f in out]

        return out

    @ImageHelper(by_frame=True, as_tuple=True)
    def tile_images(self,
                    image: Image[Optional] = tuple([]),
                    mask: Mask[Optional] = tuple([]),
                    layout: Tuple[int] = None,
                    border_value: Union[int, float] = 0.,
                    ) -> Image:
        """Tiles image stacks side by side to produced a single image. Attempts
        to do some rescaling to match intensities first, but likely will not
        produce good results for images with large differences in intensity.

        :param image: List of image stacks to be tiled.
        :param mask: List of mask stacks to be tiled.
        :param layout:
        :param border_value: Value of the default pixels.
        """
        # TODO: Add scaling of intensity and dimension
        # TODO: Add crop
        fil = sitk.TileImageFilter()
        fil.SetLayout(layout)
        fil.SetDefaultPixelValue(border_value)

        # Combine the stacks
        stacks = image + mask
        images = [cast_sitk(sitk.GetImageFromArray(s), 'sitkUInt16', True)
                  for s in stacks]

        # Rescale intensity - all calculations done in float, then cast to int
        rescale = sitk.RescaleIntensityImageFilter()
        # rescale.SetOutputMaximum(int(2 ** 16))
        # rescale.SetOutputMinimum(int(0))
        images = [rescale.Execute(i) for i in images]
        # out = cast_sitk(fil.Execute(images), 'sitkUInt16')
        out = fil.Execute(images)

        return sitk.GetArrayFromImage(out)

    @ImageHelper(by_frame=True)
    def gaussian_filter(self,
                        image: Image,
                        sigma: float = 2.5,
                        dtype: type = np.float32
                        ) -> Image:
        """
        Applies a multidimensional Gaussian filter to the image.

        :param image:
        :param sigma:
        :param dtype:

        :return:

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
        """Applies a binomial blur to the image."""
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
        """Applies a multidimensional uniform filter to the input image."""
        return ndi.uniform_filter(image, size=size,
                                  mode=mode, cval=cval)

    @ImageHelper(by_frame=True)
    def rolling_ball_background_subtract(self,
                                         image: Image,
                                         radius: float = 100,
                                         kernel: np.ndarray = None,
                                         nansafe: bool = False,
                                         return_bg: bool = False,
                                         ) -> Image:
        """
        Estimate background intensity by rolling/translating a kernel, and
        subtract from the input image.
        """
        bg = rest.rolling_ball(image, radius=radius,
                               kernel=kernel, nansafe=nansafe)
        if return_bg:
            return bg
        else:
            return image - bg

    @ImageHelper(by_frame=False)
    def n4_illumination_bias_correction(self,
                                        image: Image,
                                        mask: Mask = None,
                                        iterations: Collection[int] = 50,
                                        num_points: Collection[int] = 4,
                                        histogram_bins: int = 200,
                                        spline_order: int = 3,
                                        subsample_factor: int = 1,
                                        save_bias_field: bool = False
                                        ) -> Image:
        """
        Applies N4 bias field correction to the image. Can optionally return
        the calculated log bias field, which can be applied to the image with
        ``Process.apply_log_bias_field``.
        """
        # Check the inputs
        if (image < 1).any():
            warnings.warn('N4 correction of images with small '
                          'values can produce poor results.')
        if subsample_factor <= 1:
            warnings.warn('Faster computation can be achieved '
                          'by subsampling the original image.')

        if isinstance(iterations, int):
            iterations = [iterations] * 4  # 4 levels of correction
        else:
            assert len(iterations) == 4

        if isinstance(num_points, int):
            num_points = [num_points] * 3  # 3D Stack
        else:
            assert len(num_points) == 3

        # Set up the filter
        fil = sitk.N4BiasFieldCorrectionImageFilter()
        fil.SetMaximumNumberOfIterations(iterations)
        fil.SetNumberOfControlPoints(num_points)
        fil.SetNumberOfHistogramBins(histogram_bins)
        fil.SetSplineOrder(spline_order)

        # Load images
        img = sitk.GetImageFromArray(image)
        img = cast_sitk(img, 'sitkFloat32', cast_up=True)
        if mask is not None:
            mask = sitk.GetImageFromArray(mask)
            mask = cast_sitk(img, 'sitkUInt8')
        else:
            mask = sitk.GetImageFromArray(np.ones_like(image))
            mask = cast_sitk(mask, 'sitkUInt8')

        # Downsample images
        if subsample_factor > 1:
            shrink = sitk.ShrinkImageFilter()

            # NOTE: Image shape gets transposed, that's why the last axis
            #       factor is set to 1 instead of the first.
            factor_vector = [1 * subsample_factor for _ in image.shape]
            factor_vector[-1] = 1
            shrink.SetShrinkFactors(factor_vector)

            temp_img = shrink.Execute(img)
            temp_mask = shrink.Execute(mask)
        else:
            temp_img = img
            temp_mask = mask

        # Calculate the bias field
        _ = fil.Execute(temp_img, temp_mask)
        log_bias_field = fil.GetLogBiasFieldAsImage(img)  # Use full-res here

        if save_bias_field:
            out = cast_sitk(log_bias_field, 'sitkFloat32')
        else:
            out = img / sitk.Exp(log_bias_field)
            out = cast_sitk(out, 'sitkFloat32')

        return sitk.GetArrayFromImage(out)

    @ImageHelper(by_frame=False)
    def apply_log_bias_field(self,
                             image: Image,
                             bias_field: Image
                             ) -> Image:
        """Applies a log bias field (for example, calculated using N4
        bias illumination correction) to the input image."""
        return image / np.exp(bias_field)

    @ImageHelper(by_frame=True)
    def curvature_anisotropic_diffusion(self,
                                        image: Image,
                                        iterations: int = 5,
                                        time_step: float = 0.125,
                                        conductance: float = 1.
                                        ) -> Image:
        """Applies curvature anisotropic diffusion blurring to the image. Useful
        for smoothing out noise, while preserving the edges of objects."""
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
        Calculates gradients and inverts them on the range [0, 1],
        such that pixels close to borders have values close to 0, while
        all other pixels have values close to 1.
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
        Applies Sobel filter for edge detection. Can detect
        edges in only one dimension by using the orientation
        argument.

        TODO:
            - Could be run faster on whole stack
        """
        # Input must be in [-1, 1]
        px = get_image_pixel_type(image)
        if px == 'float':
            image = preproc.maxabs_scale(
                image.reshape(-1, 1)
            ).reshape(image.shape)
        else:
            image = util.img_as_float32(image)

        if orientation in ('h', 'horizontal'):
            sobel = filt.sobel_h(image)
        elif orientation in ('v', 'vertical'):
            sobel = filt.sobel_v(image)
        else:
            sobel = filt.sobel(image)

        return util.img_as_float32(sobel)

    @ImageHelper(by_frame=True)
    def sobel_edge_magnitude(self,
                             image: Image,
                             ) -> Image:
        """
        Similar to ``Process.sobel_edge_detection``, but returns
        the magnitude of the gradient at each pixel, without regard
        for direction.
        """
        y = ndi.sobel(image, axis=1)
        x = ndi.sobel(image, axis=0)
        return np.hypot(x, y)

    @ImageHelper(by_frame=True)
    def roberts_edge_detection(self, image: Image) -> Image:
        """Applies Roberts filter for edge detection."""
        return filt.roberts(image)

    @ImageHelper(by_frame=True)
    def recurssive_gauss_gradient(self,
                                  image: Image,
                                  sigma: float = 1.,
                                  use_direction: bool = True
                                  ) -> Image:
        """Applies recursive Gaussian filters to detect edges."""
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
        """Applies recursive Gaussian filters to detect edges
        and returns the gradient magnitude at each pixel."""
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
        Calculates an edge potential image from images with
        edges highlighted. An edge potential image has values
        close to 0 at edges, and values close to 1 else where.
        The quality of the edge potential image depends highly on the
        input image and the function/parameters used. The default
        function is 'sigmoid', which accepts two parameters to
        define the sigmoid function, alpha and beta. If you don't
        already know good values, heuristics can be used to estimate
        alpha and beta based on the minimum value along an edge (k1)
        and the average value away from an edge (k2). If no parameters
        are supplied, this function will attempt to guess.
        """
        # Cast to float first to avoid precision errors
        image = util.img_as_float32(image)

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

                k1 = np.percentile(_arr, 95)
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
        """Applies a filter to calculate the distance map of a binary
        image with objects. The distance inside objects is negative."""
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
        Rescales input image frames to match the intensity
        of a reference image. By default, the reference image
        is the first frame of the input image stack.
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
                                    return_bg: bool = False
                                    ) -> Image:
        """
        Uses discrete wavelet transformation to estimate and remove
        the background from an image.
        """
        # Pad image to even before starting
        padder = PadHelper(target='even', axis=[1, 2], mode='edge')
        image_pad = padder.pad(image)

        # Pass frames of the padded image
        # Use higher precision then downsamble later
        out = np.zeros(image_pad.shape, dtype=np.int32)
        for fr, im in enumerate(image_pad):
            bg = wavelet_background_estimate(im, wavelet, mode,
                                             level, blur)
            bg = np.asarray(bg, dtype=np.int32)

            # Remove background and ensure non-negative
            if return_bg:
                out[fr, ...] = bg
            else:
                out[fr, ...] = im - bg
            out[fr, ...][out[fr, ...] < 0] = 0

        # Undo padding and reset dtype before return
        return padder.undo_pad(out.astype(np.int16))

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
        Uses discrete wavelet transformation to estimate and remove
        noise from an image.
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
        Uses a UNet-based neural net to predict the label of each pixel in the
        input image. This function returns the probability of a specific region
        of interest, not a labeled mask.

        :param image:
        :param weight_path:
        :param roi:
        :param batch:
        :param classes:

        :return:
        """
        _roi_dict = {'background': 0, 'bg': 0, 'edge': 1,
                     'interior': 2, 'nuc': 2, 'cyto': 2}
        if isinstance(roi, str):
            try:
                roi = _roi_dict[roi]
            except KeyError:
                raise ValueError(f'Did not understand region of interest {roi}.')

        # Only import tensorflow and Keras if needed
        from celltk.utils.unet_model import FluorUNetModel

        if not hasattr(self, 'model'):
            '''NOTE: If we had mulitple colors, then image would be 4D here.
            The Pipeline isn't set up for that now, so for now the channels
            is just assumed to be 1.'''
            channels = 1
            dims = (image.shape[1], image.shape[2], channels)

            self.model = FluorUNetModel(dimensions=dims,
                                        weight_path=weight_path)

        # Pre-allocate output memory
        if batch is None or batch >= image.shape[0]:
            output = self.model.predict(image[:, :, :], roi=roi)
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a, roi=roi)
                                     for a in arrs], axis=0)

        # TODO: dtype can probably be downsampled from float32 before returning
        return output
