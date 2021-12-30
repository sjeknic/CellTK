import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from scipy.ndimage import gaussian_laplace

from cellst.operation import BaseProcess
from cellst.utils._types import Image, Mask, Track, Arr
from cellst.utils.utils import ImageHelper
from cellst.utils.operation_utils import (sliding_window_generator,
                                          shift_array, crop_array)



class Process(BaseProcess):
    @ImageHelper(by_frame=False, as_tuple=True)
    def align_by_cross_correlation(self,
                                   image: Image,
                                   mask: Mask = tuple([]),
                                   track: Track = tuple([]),
                                   align_with: str = 'image',
                                   crop: bool = True
                                   ) -> Image:
        """
        Crops images based on calculated shift from phase_cross_correlation.

        TODO:
            - Add crop working
        """
        # Get the image that aligning will be based on
        # By defaut to_align should use the first image in whatever align_with is
        to_align = locals()[align_with][0]

        # Get frame generator
        frame_generator = sliding_window_generator(to_align, overlap=1)

        # Calculate shifts using phase cross correlation
        shifts = []
        for idx, frames in enumerate(frame_generator):
            # frame_generator yields array of shape (overlap, y, x)
            shifts.append(phase_cross_correlation(frames[0, ...],
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
        return gaussian(image, sigma)

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
        return gaussian_laplace(image, sigma)

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
        bg = rolling_ball(image, radius=radius, kernel=kernel, nansafe=nansafe)
        return image - bg
