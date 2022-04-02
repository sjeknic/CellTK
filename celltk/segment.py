import warnings
import functools
from typing import Tuple, Union, Collection, Callable, List

import numpy as np
import skimage.measure as meas
import skimage.segmentation as segm
import skimage.morphology as morph
import skimage.filters as filt
import skimage.util as util
import skimage.feature as feat
import scipy.ndimage as ndi
import SimpleITK as sitk

from celltk.core.operation import BaseSegmenter
from celltk.utils._types import Image, Mask
from celltk.utils.utils import ImageHelper
from celltk.utils.operation_utils import (dilate_sitk, voronoi_boundaries,
                                          skimage_level_set, gray_fill_holes,
                                          match_labels_linear, cast_sitk,
                                          sitk_binary_fill_holes,
                                          ndi_binary_fill_holes,
                                          mask_to_seeds)


class Segmenter(BaseSegmenter):
    """
    TODO:
        - Test Taka's CellTK functions (find_boundaries, cytoring)
        - Add more cytoring functions (thres, adaptive_thres, etc.)
        - Add levelset segmentation
        - Add mahotas/watershed_distance
        - Add _threshold_dispatcher_function
        - Can Sitk be used instead of regionprops?
            (https://simpleitk.org/SPIE2019_COURSE/06_segmentation_and_shape_analysis.html)
    """
    @ImageHelper(by_frame=True)
    def label(self,
              mask: Mask,
              connectivity: int = 2
              ) -> Mask:
        """Uniiquely labels connected pixels.

        :param mask: Mask of objects to be labeled. Can be binary or greyscale.
        :param connectivity: Determines the local neighborhood around a pixel.
            Defined as the number of orthogonal steps needed to reach a pixel.

        :return: Labeled mask.
        """
        return util.img_as_uint(meas.label(mask, connectivity=connectivity))

    @ImageHelper(by_frame=True)
    def sitk_label(self,
                   mask: Mask,
                   ) -> Mask:
        """Uniquely labels connected pixels using a SimpleITK filter

        :param mask: Mask of objects to be labeled. Can be binary or greyscale.

        :return: Labeled mask
        """
        fil = sitk.ConnectedComponentImageFilter()
        msk = sitk.GetImageFromArray(mask)
        msk = cast_sitk(fil.Execute(msk), 'sitkUInt16')
        return sitk.GetArrayFromImage(msk)

    @ImageHelper(by_frame=True)
    def clean_labels(self,
                     mask: Mask,
                     min_radius: float = 3,
                     max_radius: float = 20,
                     open_size: int = 3,
                     clear_border: Union[bool, int] = True,
                     relabel: bool = False,
                     sequential: bool = False,
                     connectivity: int = 2,
                     ) -> Mask:
        """
        Applies light cleaning. Removes small, large, and border-connected
        objectes. Applies opening.

        TODO:
            - Prevent int32/64 w/o having to use img_as_uint
        """
        # Fill in holes and remove border-connected objects
        labels = gray_fill_holes(mask)

        if clear_border:
            buff = clear_border if isinstance(clear_border, (int, float)) else 3
            labels = segm.clear_border(labels, buffer_size=buff)

        # Remove small and large objects and open
        min_area, max_area = np.pi * np.array((min_radius, max_radius)) ** 2
        pos = morph.remove_small_objects(labels, min_area,
                                         connectivity=connectivity)
        neg = morph.remove_small_objects(labels, max_area,
                                         connectivity=connectivity)
        pos[neg > 0] = 0
        labels = morph.opening(pos, np.ones((open_size, open_size)))

        # Relabel the labels to separate non-contiguous objects
        if relabel:
            labels = meas.label(labels, connectivity=connectivity)

        # Make labels sequential if needed
        if sequential:
            labels = segm.relabel_sequential(labels)[0]

        return util.img_as_uint(labels)

    @ImageHelper(by_frame=True)
    def filter_objects_by_props(self,
                                mask: Mask,
                                properties: List[str],
                                limits: Collection[Tuple[float]],
                                image: Image = None,
                                ) -> Mask:
        """
        Image has to already be labeled

        :param mask:
        :param image:
        :param properties:
        :param limits:

        """
        # User must provide both low and high bound
        assert all([len(l) == 2 for l in limits])

        # Extract metrics from each region
        if 'label' not in properties:
            properties.append('label')

        rp = meas.regionprops_table(mask, image, properties=properties)

        # True in these masks are the indices for the cells to remove
        failed = [~np.logical_and(rp[prop] > lim[0], rp[prop] <= lim[1])
                  for (lim, prop) in zip(limits, properties)]
        to_remove = functools.reduce(np.add, failed).astype(bool)
        to_remove = rp['label'][to_remove]

        # Get the values and again mark indices as True
        remove_idx = functools.reduce(
            np.add,
            [np.where(mask == r, 1, 0) for r in to_remove],
        ).astype(bool)

        # Set those indices to 0 and return
        out = mask.copy()
        out[remove_idx] = 0

        return out

    @ImageHelper(by_frame=True)
    def constant_thres(self,
                       image: Image,
                       thres: Union[int, float] = 1000,
                       negative: bool = False,
                       connectivity: int = 2,
                       relative: bool = False
                       ) -> Mask:
        """
        Labels pixels above or below a constant threshold
        """
        if relative:
            assert thres <= 1 and thres >= 0
            thres = image.max() * thres

        if negative:
            test_arr = image <= thres
        else:
            test_arr = image >= thres

        # return meas.label(test_arr, connectivity=connectivity).astype(np.uint16)
        return test_arr.astype(np.uint8)

    @ImageHelper(by_frame=True)
    def adaptive_thres(self,
                       image: Image,
                       relative_thres: float = 0.1,
                       sigma: float = 50,
                       connectivity: int = 2
                       ) -> Mask:
        """
        Applies Gaussian blur to the image and selects pixels that
        are relative_thres brighter than the blurred image.
        """
        fil = ndi.gaussian_filter(image, sigma)
        fil = image > fil * (1 + relative_thres)
        return fil.astype(np.uint8)

    @ImageHelper(by_frame=True)
    def otsu_thres(self,
                   image: Image,
                   nbins: int = 256,
                   connectivity: int = 2,
                   buffer: float = 0.,
                   fill_holes: bool = False,
                   ) -> Mask:
        """
        Uses Otsu's method to determine the threshold. All pixels
        above the threshold are labeled
        """
        thres = (1 - buffer) * filt.threshold_otsu(image, nbins=nbins)
        labels = image > thres

        if fill_holes:
            # Run binary closing first to connect broken edges
            labels = morph.binary_closing(labels)
            labels = sitk_binary_fill_holes(labels)

        return labels.astype(np.uint8)

    @ImageHelper(by_frame=False)
    def multiotsu_thres(self,
                        image: Image,
                        classes: int = 2,
                        roi: Union[int, Collection[int]] = None,
                        nbins: int = 256,
                        hist: np.ndarray = None,
                        binarize: bool = False,
                        ) -> Mask:
        """"""
        thres = filt.threshold_multiotsu(image, classes, nbins,
                                         hist=hist)
        out = np.digitize(image, bins=thres).astype(np.uint8)

        # If roi is given, filter the other regions, otherwise return all
        if roi is not None:
            roi = tuple([int(roi)]) if isinstance(roi, (int, float)) else roi
            classes = np.unique(out)
            for c in classes:
                if c not in roi:
                    out[out == c] = 0

        # Clean up
        if binarize:
            out[out > 0] = 1

        return out.astype(np.uint16)

    @ImageHelper(by_frame=True)
    def li_thres(self,
                 image: Image,
                 tol: float = None,
                 init_guess: Union[float, Callable] = None,
                 connectivity: int = 2,
                 fill_holes: bool = True
                 ) -> Mask:
        """"""
        fil = sitk.LiThresholdImageFilter()
        fil.SetInsideValue(0)
        fil.SetOutsideValue(1)
        out = fil.Execute(sitk.GetImageFromArray(image))

        out = cast_sitk(out, 'sitkUInt8')
        return sitk.GetArrayFromImage(out)

    @ImageHelper(by_frame=True)
    def regional_extrema(self,
                         image: Image,
                         find: str = 'minima',
                         fully_connected: bool = True,
                         thres: float = None,
                         min_size: int = 12
                         ) -> Mask:
        """Finds regional minima/maxima within flat areas.

        TODO:
            - Is it possible to remove large objects here.
        """
        if thres:
            assert 0 <= thres and thres <= 1
            # Seems less variable than using max
            thres *= np.nanpercentile(image, 97)

        # Get the appropriate filter
        if find == 'maxima':
            if thres:
                fil = sitk.HConvexImageFilter()
                fil.SetHeight(thres)
            else: fil = sitk.RegionalMaximaImageFilter()
        else:
            if thres:
                fil = sitk.HConcaveImageFilter()
                fil.SetHeight(thres)
            else: fil = sitk.RegionalMinimaImageFilter()
        fil.SetFullyConnected(fully_connected)

        extrema = fil.Execute(sitk.GetImageFromArray(image))

        extrema = cast_sitk(extrema, 'sitkUInt16')
        conn = sitk.ConnectedComponentImageFilter()
        relab = sitk.RelabelComponentImageFilter()
        relab.SetMinimumObjectSize(min_size)
        labels = relab.Execute(conn.Execute(extrema))
        labels = cast_sitk(labels, 'sitkUInt16')
        return sitk.GetArrayFromImage(labels)

    @ImageHelper(by_frame=False)
    def sitk_filter_pipeline(self,
                             mask: Mask,
                             filters: Collection[str],
                             kwargs: Collection[dict] = [],
                             ) -> Mask:
        """"""
        # TODO: Add to process
        # TODO: Should use Stack type
        if kwargs:
            assert len(kwargs) == len(filters)
        else:
            kwargs = [{}] * len(filters)

        _sfilt = []
        for f, kw in zip(filters, kwargs):
            # Get the filter function
            try:
                _sfilt.append(getattr(sitk, f)())
            except AttributeError:
                raise AttributeError(f'Could not find SITK filter {f}')

            # Get the attributes for the filter
            for k, v in kw.items():
                try:
                    getattr(_sfilt[-1], k)(v)
                except AttributeError:
                    raise AttributeError('Could not find attribute '
                                         f'{k} in {_sfilt[-1]}')

        # Apply to all the frames individually
        out = np.zeros_like(mask)
        for n, fr in enumerate(mask):
            _im = sitk.GetImageFromArray(fr)
            for fil in _sfilt:
                _im = fil.Execute(_im)

            out[n, ...] = sitk.GetArrayFromImage(_im)

        return out

    @ImageHelper(by_frame=True)
    def random_walk_segmentation(self,
                                 image: Image,
                                 seeds: Mask = None,
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
              random_walk not expand too much when labeling?
        """
        # Generate seeds
        if seeds is None:
            seeds = meas.label(image >= seed_thres)

        if seed_min_size is not None:
            seeds = morph.remove_small_objects(seeds, seed_min_size)

        # Anisotropic diffusion from each seed
        probs = segm.random_walker(image, seeds,
                                   beta=beta, tol=tol,
                                   return_full_prob=True)

        # Label seeds based on probability threshold
        mask = probs >= seg_thres
        out = np.empty(mask.shape).astype(np.uint16)
        for p in range(probs.shape[0]):
            # Where mask is True, label those pixels with p
            np.place(out, mask[p, ...], p)

        return util.img_as_uint(out)

    @ImageHelper(by_frame=True)
    def agglomeration_segmentation(self,
                                   image: Image,
                                   seeds: Mask = None,
                                   agglom_min: float = 0.7,
                                   agglom_max: float = None,
                                   compact: float = 100,
                                   seed_thres: float = 0.975,
                                   seed_min_size: float = 12,
                                   steps: int = 50,
                                   connectivity: int = 2
                                   ) -> Mask:
        """
        Starts from a seed mask determined by a constant threshold. Then
        incrementally uses watershed to connect neighboring pixels to the
        seed.

        Similar to, but likely faster than, random walk segmentation.

        TODO:
            - agglom_min and agglom_max should be set as a fraction of the image values
        """
        # Candidate pixels and percentiles can be set on 3D stack
        if agglom_max is None: agglom_max = np.nanmax(image)
        percs = np.linspace(agglom_max, agglom_min, steps)

        # Generate seeds based on constant threshold
        if seeds is None:
            seeds = image >= seed_thres

        if seed_min_size is not None:
            seeds = morph.remove_small_objects(seeds, seed_min_size,
                                               connectivity=2)

        # Iterate through pixel values and add using watershed
        _old_perc = agglom_max
        for _perc in percs:
            # Candidate pixels are between _perc and the last _perc value
            cand_mask = np.logical_and(image > _perc,
                                       image <= _old_perc)
            # Keep seeds in the mask as well
            cand_mask = np.logical_or(seeds > 0, cand_mask > 0)

            # Watershed and save as seeds for the next iteration
            seeds = segm.watershed(image, seeds, mask=cand_mask,
                                   watershed_line=True, compactness=compact)
            _old_perc = _perc

        return util.img_as_uint(seeds)

    @ImageHelper(by_frame=True)
    def watershed_ift_segmentation(self,
                                   image: Image,
                                   seeds: Mask = None,
                                   seed_thres: float = 0.975,
                                   seed_min_size: float = 12,
                                   connectivity: int = 2
                                   ) -> Mask:
        """
        Wrapper for scipy.ndimage.watershed_ift.
        TODO:
            - Accept pre-made seed mask from a different function
        """
        # Generate seeds based on constant threshold
        if seeds is None:
            seeds = meas.label(image >= seed_thres)

        if seed_min_size is not None:
            seeds = morph.remove_small_objects(seeds, seed_min_size, connectivity=2)

        # Convert background pixels to negative
        seeds = seeds.astype(np.uint16)  # convert to signed integers
        # seeds[seeds == 0] = -1
        # Search area is equivalent to connectivity = 2
        struct = np.ones((3, 3)).astype(np.uint8)

        # Watershed and remove negatives
        out = ndi.watershed_ift(util.img_as_uint(image), seeds, struct)
        out[out < 0] = 0

        return util.img_as_uint(out)

    @ImageHelper(by_frame=True)
    def chan_vese_dense_levelset(self,
                                 image: Image,
                                 seeds: Mask,
                                 iterations: int = 70,
                                 smoothing: float = 0,
                                 curve_weight: float = 1,
                                 area_weight: float = 1,
                                 lambda1: float = 1,
                                 lambda2: float = 1,
                                 epsilon: float = 1,
                                 ) -> Mask:
        """"""
        # Set up the filter
        fil = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
        fil.SetNumberOfIterations(iterations)
        fil.SetReinitializationSmoothingWeight(smoothing)
        fil.SetAreaWeight(area_weight)
        fil.SetCurvatureWeight(curve_weight)
        fil.SetLambda1(lambda1)
        fil.SetLambda2(lambda2)
        fil.SetEpsilon(epsilon)

        # Get the images and execute
        img = sitk.GetImageFromArray(image)
        img = cast_sitk(img, 'sitkFloat32', cast_up=True)
        msk = sitk.GetImageFromArray(seeds)
        msk = cast_sitk(msk, 'sitkFloat32', cast_up=True)
        out = fil.Execute(msk, img)

        return sitk.GetArrayFromImage(cast_sitk(out, 'sitkUInt16'))

    @ImageHelper(by_frame=True)
    def morphological_acwe(self,
                           image: Image,
                           seeds: Mask = 'checkerboard',
                           iterations: int = 10,  # TODO: Set appr. value
                           smoothing: int = 1,
                           lambda1: float = 1,
                           lambda2: float = 1,
                           connectivity: int = 1,
                           thres: float = None,
                           keep_labels: bool = True,
                           clean_before_match: bool = True
                           ) -> Mask:
        """
        Uses morphological_chan_vese to segment objects, followed by a
        voronoi calculation to separate them.

        Args:
        thres - if set, only considers values in image > thres
        keep_labels - uses linear assignment to transfer seed labels
        clean_before_match - apply simple cleaning to masks before match

        TODO:
            - Add option to draw voronoi boundaries after each iteration
            - Should thresholding happen before or after morph?
        """
        # Get level_set mask if needed - dont use voronoi boundaries
        if not isinstance(seeds, np.ndarray):
            seeds = skimage_level_set(image.shape, levelset=seeds)
            vor_mask = np.zeros(image.shape, dtype=bool)
        else:
            # Get Voronoi boundaries to use to separate objects
            vor_mask = voronoi_boundaries(seeds, thin=False)

        # Apply threshold to image. Should this happen before or after???
        if thres:
            image = np.where(image > thres, image, 0)

        # Propagate shapes
        regions = segm.morphological_chan_vese(image, iterations, seeds,
                                               smoothing, lambda1, lambda2)

        # Remove voronoi boundaries and label objects
        regions[vor_mask] = 0
        regions = meas.label(regions, connectivity=connectivity)

        # # If seed mask is provided, transfer labels
        if keep_labels and isinstance(seeds, np.ndarray):
            # Optional cleaning to remove small objects and whatnot
            if clean_before_match:
                regions = self.clean_labels.__wrapped__(self, regions)
            regions = match_labels_linear(seeds, regions)

        return util.img_as_uint(regions)

    @ImageHelper(by_frame=True)
    def morphological_geodesic_active_contour(self,
                                              image: Image,
                                              seeds: Mask = 'checkerboard',
                                              iterations: int = 50,  # TODO: Set appr. value
                                              smoothing: int = 1,
                                              threshold: float = 'auto',
                                              balloon: float = 0
                                              ) -> Mask:
        """
        Should run skimage.segmentation.morphological_geodesic_active_contour

        TODO:
            - Add preprocess options - sobel, inverse gaussian
        """
        return segm.morphological_geodesic_active_contour(image, iterations,
                                                          seeds, smoothing,
                                                          threshold, balloon)

    @ImageHelper(by_frame=True)
    def convex_hull_object(self,
                           mask: Mask,
                           connectivity: int = 2,
                           ) -> Mask:
        """"""
        # Binarize image
        if mask.max() > 1: mask = mask.astype(bool)

        return morph.convex_hull_object(mask)

    @ImageHelper(by_frame=True)
    def canny_edge_segmentation(self,
                                image: Image,
                                sigma: float = 1.,
                                low_thres: float = None,
                                high_thres: float = None,
                                use_quantiles: bool = False,
                                fill_holes: bool = False
                                ) -> Mask:
        """"""
        out = feat.canny(image, sigma=sigma, low_threshold=low_thres,
                         high_threshold=high_thres,
                         use_quantiles=use_quantiles)

        if fill_holes:
            out = sitk_binary_fill_holes(out)

        return util.img_as_uint(out)

    @ImageHelper(by_frame=True)
    def find_boundaries(self,
                        mask: Mask,
                        connectivity: int = 2,
                        mode: str = 'inner',
                        keep_labels: bool = False
                        ) -> Mask:
        """
        Outlines the objects in mask and preserves labels.

        if not keep_labels - don't preserve labels
        """
        boundaries = segm.find_boundaries(mask, connectivity=connectivity,
                                          mode=mode)

        if keep_labels:
            return np.where(boundaries, mask, 0)
        else:
            return boundaries

    @ImageHelper(by_frame=True)
    def dilate_to_cytoring_celltk(self,
                                  mask: Mask,
                                  ringwidth: int = 1,
                                  margin: int = 1
                                  ) -> Mask:
        """
        Copied directly from CellTK. Should dilate out from nuclear mask
        to create cytoplasmic ring.

        NOTE:
            - I think this is done in greyscale, so labels should be preserved.
            - I think ringwidth is the amount to expand the labels
            - I think margin is dist. between the nuclear mask and the cytoring
            - No clue why multiple rounds of dilation are used.

        TODO:
            - Re-write this function. Needs to be consistent
            - There is another function in CellTK that uses Voronoi expansion
              to set a buffer between adjacent cytorings. Copy that functionality
              here.
            - Add cytoring_above_thres, cytoring_above_adaptive_thres,
              cytoring_above_buffer
        """
        dilated_nuc = dilate_sitk(mask.astype(np.int32), ringwidth)

        # TODO: Replace with np.where(mask == 0, 0, comp_dilated_nuc)??
        comp_dilated_nuc = 1e4 - mask
        comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0

        #  TODO: Why is the dilation done twice?
        comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), ringwidth)
        # TODO: See replacement above.
        comp_dilated_nuc = 1e4 - comp_dilated_nuc
        comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
        # TODO: Not sure what this is for
        dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0

        # TODO: This is for adding the margin. Why is the if/else needed?
        if margin == 0:
            antinucmask = mask
        else:
            antinucmask = dilate_sitk(np.int32(mask), margin)
        dilated_nuc[antinucmask.astype(bool)] = 0

        return util.img_as_uint(np.uint16)

    @ImageHelper(by_frame=True)
    def expand_to_cytoring(self,
                           mask: Mask,
                           distance: float = 1,
                           margin: int = 0
                           ) -> Mask:
        """
        Creating a cytoring using skimage functions

        TODO:
            - need to implement margin
        """
        # Expand initial seeds before applying expansion
        if margin:
            mask = segm.expand_labels(mask, margin)

        out = segm.expand_labels(mask, distance)

        return out - mask

    @ImageHelper(by_frame=True)
    def remove_nuc_from_cyto(self,
                             cyto_mask: Mask,
                             nuc_mask: Mask,
                             val_match: bool = False,
                             erosion: bool = False
                             ) -> Mask:
        """
        Removes nuclei from a cytoplasmic mask
        """
        if val_match:
            new_cyto_mask = np.where(cyto_mask == nuc_mask, 0, cyto_mask)
        else:
            new_cyto_mask = np.where(nuc_mask, 0, cyto_mask)

        if erosion:
            binary_img = new_cyto_mask.astype(bool)
            eroded_img = morph.binary_erosion(binary_img)
            new_cyto_mask = np.where(eroded_img, new_cyto_mask, 0)

        return util.img_as_uint(new_cyto_mask)

    @ImageHelper(by_frame=True)
    def binary_fill_holes(self,
                          mask: Mask,
                          fill_border: bool = True,
                          iterations: Union[bool, int] = False,
                          kernel_radius: int = 4,
                          max_length: int = 45,
                          in_place: bool = True,
                          method: str = 'sitk',
                          **kwargs
                          ) -> Mask:
        """
        kwargs get used to set attributes on the sitk filters that were used

        TODO:
            - Write a greyscale version (will work better on the borders,
                but will need a new _close_border_holes)
        """
        if method == 'sitk':
            return sitk_binary_fill_holes(mask, fill_border, iterations,
                                          kernel_radius, max_length,
                                          in_place, **kwargs)
        elif method == 'ndi':
            return ndi_binary_fill_holes(mask, fill_border, kernel_radius,
                                         max_length, in_place)

    @ImageHelper(by_frame=True)
    def level_set_mask(self,
                       image: Image,
                       levelset: str = 'checkerboard',
                       size: Union[float, int] = None,
                       center: Tuple[int] = None,
                       label: bool = False
                       ) -> Mask:
        """
        Wrapper for levelset functions in skimage.segmentation

        size refers to square_size for checkerboard or radius for disk
        """
        mask = skimage_level_set(image.shape, levelset, size, center)

        if label:
            mask = meas.label(mask)

        return mask

    @ImageHelper(by_frame=True)
    def shape_detection_level_set(self,
                                  edge_potential: Image,
                                  initial_level_set: Mask,
                                  curvature: float = 1,
                                  propagation: float = 1,
                                  iterations: int = 1000,
                                  ) -> Mask:
        """"""
        # Set up filter
        fil = sitk.GeodesicActiveContourLevelSetImageFilter()
        fil.SetCurvatureScaling(curvature)
        fil.SetPropagationScaling(propagation)
        fil.SetNumberOfIterations(iterations)
        # Check edge potential format
        if edge_potential.max() > 1 or edge_potential.min() < 0:
            warnings.warn('Edge potential image seems poorly formatted. '
                          'Should be 0 at edges and 1 elsewhere.', UserWarning)

        init = sitk.GetImageFromArray(initial_level_set)
        edge = sitk.GetImageFromArray(edge_potential)
        init = cast_sitk(init, 'sitkFloat32', cast_up=True)
        edge = cast_sitk(edge, 'sitkFloat32')

        out = cast_sitk(fil.Execute(edge, init), 'sitkUInt16')
        return sitk.GetArrayFromImage(out)

    @ImageHelper(by_frame=True)
    def threshold_level_set(self,
                            image: Image,
                            initial_level_set: Mask,
                            lower_thres: float = None,
                            upper_thres: float = None,
                            propagation: float = 1,
                            curvature: float = 1,
                            iterations: int = 1000,
                            max_rms_error: float = 0.02,
                            ) -> Mask:
        """"""
        # Set up the level set filter
        fil = sitk.ThresholdSegmentationLevelSetImageFilter()
        fil.SetCurvatureScaling(curvature)
        fil.SetPropagationScaling(propagation)
        fil.SetNumberOfIterations(iterations)
        fil.SetMaximumRMSError(max_rms_error)
        if lower_thres: fil.SetLowerThreshold(lower_thres)
        if upper_thres: fil.SetUpperThreshold(upper_thres)

        # Cast inputs and apply the change
        init = sitk.GetImageFromArray(initial_level_set)
        img = sitk.GetImageFromArray(image)
        init = cast_sitk(init, 'sitkFloat32', cast_up=True)
        img = cast_sitk(img, 'sitkFloat32', cast_up=True)

        out = cast_sitk(fil.Execute(init, img), 'sitkUInt16')
        return sitk.GetArrayFromImage(out)

    @ImageHelper(by_frame=True)
    def confidence_connected(self,
                             image: Image,
                             seeds: Mask,
                             radius: int = 1,
                             multiplier: float = 4.5,
                             iterations: int = 10
                             ) -> Mask:
        """"""
        # Set up filter
        fil = sitk.ConfidenceConnectedImageFilter()
        fil.SetMultiplier(multiplier)
        fil.SetInitialNeighborhoodRadius(radius)
        fil.SetNumberOfIterations(iterations)

        # Get seed points from the provided mask
        seeds = list(zip(*np.where(seeds)))
        seeds = [(int(s[0]), int(s[1])) for s in seeds]
        fil.SetSeedList(seeds)

        # Execute and return
        img = fil.Execute(sitk.GetImageFromArray(image))
        img = cast_sitk(img, 'sitkUInt16')
        return sitk.GetArrayFromImage(img)

    @ImageHelper(by_frame=True)
    def fast_marching_level_set(self,
                                image: Image,
                                seeds: Mask,
                                n_points: int = 0
                                ) -> Mask:
        """"""
        # TODO: Add using regular seeds instead of a mask
        if n_points:
            seeds = self.regular_seeds(image, n_points)

        # Set up filter with trial points from seeds
        # pts = list(zip(*np.where(seeds)))
        pts = mask_to_seeds(seeds, output='points')
        fil = sitk.FastMarchingImageFilter()
        for pt in pts:
            fil.AddTrialPoint((int(pt[0]), int(pt[1]), int(0)))

        # Cast inputs and run
        img = sitk.GetImageFromArray(image)
        img = fil.Execute(img)
        img = cast_sitk(img, 'sitkFloat32')
        return sitk.GetArrayFromImage(img)

    @ImageHelper(by_frame=True)
    def watershed_from_markers(self,
                               image: Image,
                               mask: Mask,
                               watershed_line: bool = True,
                               remove_large: bool = True
                               ) -> Mask:
        """"""
        fil = sitk.MorphologicalWatershedFromMarkersImageFilter()
        fil.SetMarkWatershedLine(watershed_line)

        # Must be scalar type for watershed filter
        img = sitk.GetImageFromArray(image)
        msk = sitk.GetImageFromArray(mask)
        img = cast_sitk(img, 'sitkUInt16', cast_up=True)
        msk = cast_sitk(msk, 'sitkUInt16', cast_up=True)

        out = util.img_as_uint(sitk.GetArrayFromImage(fil.Execute(img, msk)))

        if remove_large:
            # The background sometimes gets segmented, so remove it
            objects, counts = np.unique(out, return_counts=True)
            # Anything larger than 1/5 the image is masked
            count_mask = counts >= 0.2 * np.prod(image.shape)
            to_remove = objects[count_mask]
            if to_remove.any():
                for t in to_remove:
                    out[out == t] = 0

        return out

    @ImageHelper(by_frame=True)
    def morphological_watershed(self,
                                image: Image,
                                watershed_line: bool = True
                                ) -> Mask:
        """"""
        fil = sitk.MorphologicalWatershedImageFilter()
        fil.SetMarkWatershedLine(watershed_line)

        img = sitk.GetImageFromArray(image)
        img = fil.Execute(img)
        img = cast_sitk(img, 'sitkUInt16')

        return sitk.GetArrayFromImage(img)

    @ImageHelper(by_frame=False, as_tuple=True)
    def label_by_voting(self,
                        mask: Mask,
                        label_undecided: int = 0
                        ) -> Mask:
        """"""
        # Set up the vector
        fil = sitk.LabelVotingImageFilter()
        fil.SetLabelForUndecidedPixels(label_undecided)

        # Get the images
        imgs = [sitk.GetImageFromArray(m) for m in mask]
        imgs = [cast_sitk(i, 'sitkUInt16') for i in imgs]
        if len(imgs) > 5:
            mask = fil.Execute(imgs)
        else:
            mask = fil.Execute(*imgs)

        return sitk.GetArrayFromImage(mask)

    @ImageHelper(by_frame=False)
    def unet_predict(self,
                     image: Image,
                     weight_path: str = 'celltk/config/unet_example_cell_weights.hdf5',
                     roi: Union[int, str] = 2,
                     batch: int = None,
                     classes: int = 3,
                     ) -> Image:
        """
        This should just call the Process version __wrapped__
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
        # TODO: Incorporate the batch here.
        if batch is None:
            output = self.model.predict(image[:, :, :], roi=roi)
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a, roi=roi)
                                     for a in arrs], axis=0)

        # TODO: dtype can probably be downsampled from float32 before returning
        return output

    @ImageHelper(by_frame=False)
    def misic_predict(self,
                      image: Image,
                      model_path: str = 'external/misic/MiSiCv2.h5',
                      weight_path: str = None,
                      batch: int = None,
                      ) -> Mask:
        """Wrapper for implementation of Misic paper.

        TODO:
            - Make custom structure to speed up calculations
        """
        # Only import tensorflow if needed
        from celltk.utils.unet_model import MisicModel
        self.model = MisicModel(model_path)

        # Use model for predictions
        if batch is None:
            output = self.model.predict(image[:, :, :], )
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a) for a in arrs],
                                    axis=0)

        return output
