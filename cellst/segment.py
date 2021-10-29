from typing import Collection, Tuple

import numpy as np
from skimage.measure import label
from skimage.segmentation import (clear_border, random_walker,
                                  relabel_sequential, watershed)
from skimage.morphology import remove_small_objects, opening
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

from cellst.operation import Operation
from cellst.utils._types import Image, Mask
from cellst.utils.utils import image_helper
from cellst.utils.operation_utils import remove_small_holes_keep_labels


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
                     relabel: bool = False,
                     sequential: bool = False,
                     connectivity: int = 2,
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

            # Fill in holes and remove border-connected objects
            labels = remove_small_holes_keep_labels(ma, np.pi * min_radius ** 2)
            labels = clear_border(labels, buffer_size=2)

            # Remove small and large objects and open
            pos = remove_small_objects(labels, min_area,
                                       connectivity=connectivity)
            neg = remove_small_objects(labels, max_area,
                                       connectivity=connectivity)
            pos[neg > 0] = 0
            labels = opening(pos, np.ones((open_size, open_size)))

            # Relabel the labels to separate non-contiguous objects
            if relabel:
                labels = label(labels, connectivity=connectivity)

            # Make labels sequential if needed
            if sequential:
                labels = relabel_sequential(labels)[0]

            out[fr, ...] = labels

        return out

    # TODO: Should these methods actually be static? What's the benefit?
    # TODO: Consistent kwarg naming scheme
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
            out[fr, ...] = label(test_arr[fr, ...],
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
            out[fr, ...] = label(filt, connectivity=connectivity)

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
            out[fr, ...] = label(image[fr, ...] > thres,
                                 connectivity=connectivity)

        return out

    @staticmethod
    @image_helper
    def random_walk_segmentation(image: Image,
                                 seed_thres: float = 0.99,
                                 seed_min_size: float = 12,
                                 beta: float = 80,
                                 tol: float = 0.01,
                                 seg_thres: float = 0.85,
                                 ) -> Mask:
        """
        Uses random aniostropic diffusion from seeds determined
        by a constant threshold to assign each pixel in the image.
        NOTE: This function is similar to, but slower than, agglomerations.

        TODO:
            - Could setting the image values all to 0 help the
              random_walk not expand too much.
        """
        out = np.empty(image.shape).astype(np.uint16)
        seeds = image >= seed_thres
        for fr in range(image.shape[0]):
            # Generate seeds
            seed = label(seeds[fr, ...])
            seed = remove_small_objects(seed, seed_min_size)

            # Anisotropic diffusion from each seed
            probs = random_walker(image[fr, ...], seed,
                                  beta=beta, tol=tol,
                                  return_full_prob=True)

            # Using probabilites because the ranom-walk expands
            # the masks too much.
            mask = probs >= seg_thres
            for p in range(probs.shape[0]):
                # Where mask is True, label those pixels with p
                np.place(out[fr, ...], mask[p, ...], p)

        return out

    @staticmethod
    @image_helper
    def agglomeration_segmentation(image: Image,
                                   agglom_min: float = 0.7,
                                   agglom_max: float = None,
                                   compact: float = 100,
                                   seed_thres: float = 0.975,
                                   seed_min_size: float = 12,
                                   steps: int = 100,
                                   connectivity: int = 2
                                   ) -> Mask:
        """
        Starts from a seed mask determined by a constant threshold. Then
        incrementally uses watershed to connect neighboring pixels to the
        seed.

        Similar to, but likely faster than, random walk segmentation.

        TODO:
            - This function could be able to take in a pre-made seed mask
              This might be challenging, because I don't think I can currently
              make any of (Image, Mask, Track, Arr) optional.
        """
        # Candidate pixels and percentiles can be set on 3D stack
        if agglom_max is None: agglom_max = np.nanmax(image)
        percs = np.linspace(agglom_max, agglom_min, steps)

        out = np.empty(image.shape).astype(np.uint16)
        for fr in range(image.shape[0]):
            # Generate seeds based on constant threshold
            seeds = image[fr, ...] > seed_thres
            seeds = remove_small_objects(seeds, seed_min_size, connectivity=2)

            # Iterate through pixel values and add using watershed
            _old_perc = agglom_max
            for _perc in percs:
                # Candidate pixels are between _perc and the last _perc value
                cand_mask = np.logical_and(image[fr, ...] > _perc,
                                           image[fr, ...] <= _old_perc)
                # Keep seeds in the mask as well
                cand_mask = np.logical_or(seeds > 0, cand_mask > 0)

                # Watershed and save as seeds for the next iteration
                seeds = watershed(image[fr, ...], seeds, mask=cand_mask,
                                  watershed_line=True, compactness=compact)
                _old_perc = _perc

            out[fr, ...] = label(seeds, connectivity=connectivity)

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
