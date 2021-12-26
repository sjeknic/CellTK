import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from scipy.ndimage import gaussian_laplace

from cellst.operation import BaseProcess
from cellst.utils._types import Image, Mask, Track, Arr
from cellst.utils.utils import ImageHelper
from cellst.utils.operation_utils import sliding_window_generator, shift_array



class Process(BaseProcess):
    def align_by_cross_correlation(self,
                                   image: Image,
                                   mask: Mask = None,
                                   track: Track = None,
                                   array: Arr = None,
                                   align_with: str = 'image',
                                   apply_to_all: bool = True,
                                   crop: bool = True
                                   ) -> Image:
        """
        Crops images based on calculated shift from phase_cross_correlation.

        TODO:
            - Remove need to have array in args. Needs custom Process.run_operation
            - Add crop working
            - Add applying to all inputs
        """
        # Get the image that aligning will be based on
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
        crop_idx = (np.argmax(np.abs(cumulative[:, 0])),
                    np.argmax(np.abs(cumulative[:, 1])))
        crop_vals = cumulative[crop_idx[0], 0], cumulative[crop_idx[1], 1]

        # Store outputs to return
        out = np.empty_like(to_align)
        out[0, ...] = to_align[0, ...]
        for idx, (cumul, frame) in enumerate(zip(cumulative, to_align[1:])):
            out[idx + 1, ...] = shift_array(frame, cumul, 0, crop_vals)

        return Image, out

    @ImageHelper(by_frame=False)
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
            - SimpleITK implementation should be faseter
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
