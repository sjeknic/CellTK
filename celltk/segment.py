import warnings
import functools
from typing import Tuple, Union, Collection, List

import numpy as np
import skimage.measure as meas
import skimage.segmentation as segm
import skimage.morphology as morph
import skimage.filters as filt
import skimage.util as util
import skimage.feature as feat
import scipy.ndimage as ndi
import SimpleITK as sitk
import pkg_resources as pkg

from celltk.core.operation import BaseSegment
from celltk.utils._types import Image, Mask, Stack, Optional
from celltk.utils.utils import ImageHelper
from celltk.utils.operation_utils import (dilate_sitk, voronoi_boundaries,
                                          skimage_level_set, gray_fill_holes,
                                          match_labels_linear, cast_sitk,
                                          sitk_binary_fill_holes,
                                          ndi_binary_fill_holes,
                                          mask_to_seeds)


class Segment(BaseSegment):
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
        """Uniquely labels connected pixels.

        :param mask: Mask of objects to be labeled. Can be binary or greyscale.
        :param connectivity: Determines the local neighborhood around a pixel.
            Defined as the number of orthogonal steps needed to reach a pixel.

        :return: Labeled mask.
        """
        return util.img_as_uint(meas.label(mask, connectivity=connectivity))

    @ImageHelper(by_frame=True)
    def sitk_label(self,
                   mask: Mask,
                   min_size: float = None
                   ) -> Mask:
        """Uniquely labels connected pixels. Also removes connected objects below
        a specified size.

        :param mask: Mask of objects to be labeled. Can be binary or greyscale.
        :param min_size: If given, all objects smaller than min_size are removed.

        :return: Labeled mask

        TODO:
            - These functions are confusing and only one should be kept.
        """
        fil = sitk.ConnectedComponentImageFilter()
        msk = sitk.GetImageFromArray(mask)
        msk = cast_sitk(fil.Execute(msk), 'sitkUInt16')
        if min_size:
            fil2 = sitk.RelabelComponentImageFilter()
            fil2.SetMinimumObjectSize(min_size)
            msk = fil2.Execute(msk)
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
        Applies light cleaning intended for relatively small mammalian nuclei.
        Fills holes, removes small, large, and border-connected objectes, and
        then applies morphological opening.

        :param mask: Mask of uniquely labeled cell nuclei
        :param min_radius: Objects smaller than a circle with radius min_radius
            are removed.
        :param max_radius: Objects larger than a circle with radius max_radius
            are removed.
        :param open_size: Side length of the footprint used for the morphological
            opening operation.
        :param clear_border: If True or int, adny object connected to the border
            is removed. If an integer is given, objects that many pixels from the
            border are also removed.
        :param relabel: If True, objects are relabeled after cleaning. This can
            help separate objects that were connected before cleaning, but are no
            longer connected.
        :param sequential: If True, objects are relabeled with sequential labels.
        :param connectivity: Determines the local neighborhood around a pixel for
            defining connected objects. Connectivity is defined as the number of
            orthogonal steps needed to reach a pixel.

        :return: Mask with cleaned labels

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
        """Removes objects from a labeled image based on shape properties.
        Acceptable properties are ones available for
        skimage.measure.regionprops.

        :param mask:
        :param image:
        :param properties:
        :param limits:

        :return:

        TODO:
            - Add using an intensity image too
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
        out = mask.copy()
        if len(to_remove):
            remove_idx = functools.reduce(
                np.add,
                [np.where(mask == r, 1, 0) for r in to_remove],
            ).astype(bool)

            # Set those indices to 0 and return
            out[remove_idx] = 0

        return out

    @ImageHelper(by_frame=True)
    def constant_thres(self,
                       image: Image,
                       thres: Union[int, float] = 1000,
                       negative: bool = False,
                       connectivity: int = 2,
                       relative: bool = False
                       ) -> Mask[np.uint8]:
        """Labels pixels above or below a threshold value.

        :param image:
        :param thres:
        :param negative:
        :param connectivity:
        :param relative:

        :return: Labeled binary mask
        """
        if relative:
            assert thres <= 1 and thres >= 0
            thres = image.max() * thres

        if negative:
            test_arr = image <= thres
        else:
            test_arr = image >= thres

        return test_arr

    @ImageHelper(by_frame=True)
    def adaptive_thres(self,
                       image: Image,
                       relative_thres: float = 0.1,
                       sigma: float = 50,
                       connectivity: int = 2
                       ) -> Mask[np.uint8]:
        """Applies Gaussian blur to the image and marks pixels that
        are brighter than the blurred image by a specified threshold.

        :param image:
        :param relative_thres:
        :param sigma:
        :param connectivity:

        :return:
        """
        fil = ndi.gaussian_filter(image, sigma)
        fil = image > fil * (1 + relative_thres)
        return fil

    @ImageHelper(by_frame=True)
    def otsu_thres(self,
                   image: Image,
                   nbins: int = 256,
                   connectivity: int = 2,
                   buffer: float = 0.,
                   fill_holes: bool = False,
                   ) -> Mask[np.uint8]:
        """Uses Otsu's method to determine the threshold and labels all pixels
        above the threshold.

        :param image:
        :param nbins:
        :param connectivity:
        :param buffer:
        :param fill_holes:

        :return:
        """
        thres = (1 - buffer) * filt.threshold_otsu(image, nbins=nbins)
        labels = image > thres

        if fill_holes:
            # Run binary closing first to connect broken edges
            labels = morph.binary_closing(labels)
            labels = sitk_binary_fill_holes(labels)

        return labels

    @ImageHelper(by_frame=False)
    def multiotsu_thres(self,
                        image: Image,
                        classes: int = 2,
                        roi: Union[int, Collection[int]] = None,
                        nbins: int = 256,
                        hist: np.ndarray = None,
                        binarize: bool = False,
                        ) -> Mask[np.uint8]:
        """Applies Otsu's thresholding with multiple classes. By default,
        returns a mask with all classes included, but can be limited to
        only returning some of the classes.

        :param image:
        :param classes:
        :param roi:
        :param nbins:
        :param hist:
        :param binarize:

        :return:
        """
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

        return out.astype(np.uint8)

    @ImageHelper(by_frame=True)
    def li_thres(self,
                 image: Image,
                 inside_val: int = 0,
                 outside_val: int = 1
                 ) -> Mask[np.uint8]:
        """Applies Li's thresholding method.

        :param image:
        :param inside_val:
        :param outside_val:

        :return:

        """

        fil = sitk.LiThresholdImageFilter()
        fil.SetInsideValue(inside_val)
        fil.SetOutsideValue(outside_val)
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
        Generally requires an image with little noise.

        :param image:
        :param find:
        :param fully_connected:
        :param thres:
        :param min_size:

        :return: Mask with regional extrema labeled
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
        """Applies arbitrary filters from the SimpleITK package.
        Any image filter in SimpleITK, and most arguments for those
        filters, can be used.

        :param mask:
        :param filters:
        :param kwargs:

        :return:
        """
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
        """Uses random aniostropic diffusion from given seeds
        to assign each pixel in the image to a specific seed.

        :param image:
        :param seeds:
        :param seed_thres:
        :param seed_min_size:
        :param beta:
        :param tol:
        :param seg_thres:

        :return:

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
        incrementally uses watershed to connect neighboring pixels to each
        seed.

        :param image:
        :param seeds:
        :param agglom_min:
        :param agglom_max:
        :param compact:
        :param seed_thres:
        :param seed_min_size:
        :param steps:
        :param connectivity:

        :return:

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
        Applies watershed from the given seeds using the image foresting
        transform algorithm.

        :param image:
        :param seeds:
        :param seed_thres:
        :param seed_min_size:
        :param connectivity:

        :return:

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
        """Calculates the Chan-Vese level set from initial seeds.
        Similar to ``Segment.morphological_acwe``, but more
        customizable.

        :param image:
        :param mask:
        :param iterations:
        :param smoothing:
        :param curve_weight:
        :param area_weight:
        :param lambda1:
        :param lambda2:
        :param epsilon:

        :return:
        """
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
                           seeds: Mask[Optional] = 'checkerboard',
                           iterations: int = 10,
                           smoothing: int = 1,
                           lambda1: float = 1,
                           lambda2: float = 1,
                           connectivity: int = 1,
                           thres: float = None,
                           voronoi: bool = False,
                           keep_labels: bool = True,
                           clean_before_match: bool = True
                           ) -> Mask:
        """
        Uses morphological active contours without edges (ACWE) algorithm
        to segment objects. Optionally, applies a vornoi mask after segmentation
        to separate objects.

        :param image:
        :param seeds:
        :param iterations:
        :param smoothing:
        :param lambda1:
        :param lambda2:
        :param connectivity:
        :param thres:
        :param voronoi:
        :param keep_labels:
        :param clean_before_match:

        :return:

        TODO:
            - Should thresholding happen before or after morph?
        """
        # Get level_set mask if needed - dont use voronoi boundaries
        if not isinstance(seeds, np.ndarray):
            seeds = skimage_level_set(image.shape, levelset=seeds)
            vor_mask = np.zeros(image.shape, dtype=bool)
        else:
            # Get Voronoi boundaries to use to separate objects
            if voronoi:
                vor_mask = voronoi_boundaries(seeds, thin=False)
            else:
                vor_mask = np.zeros(image.shape, dtype=bool)

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
                                              seeds: Mask[Optional] = 'checkerboard',
                                              iterations: int = 10,
                                              smoothing: int = 1,
                                              threshold: float = 'auto',
                                              balloon: float = 0
                                              ) -> Mask:
        """
        Applies geodesic active contours with morphological operators to segment objects.
        Light wrapper for ``skimage.segmentation.morphological_geodesic_active_contour``.

        :param image:
        :param seeds:
        :param iterations:
        :param smoothing:
        :param threshold:
        :param balloon:

        :return:

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
        """
        Computes the convex hull of each object found in a binary
        mask. If mask is not binary already, will be binarized.

        :param mask:
        :param connectivity:

        :return:
        """
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
        """Uses a Canny filter to find edges in the image.

        :param image:
        :param sigma:
        :param low_thresh:
        :param high_thres:
        :param use_quantiles:
        :param fill_holes:

        :return:
        """
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
                        mode: str = 'thick',
                        keep_labels: bool = True
                        ) -> Mask:
        """
        Returns the outlines of the objects in the mask, optionally
        preserving the labels.

        :param mask:
        :param connectivity:
        :param mode:
        :param keep_labels:

        :return:
        """
        boundaries = segm.find_boundaries(mask, connectivity=connectivity,
                                          mode=mode)

        if keep_labels:
            return np.where(boundaries, mask, 0)
        else:
            return boundaries

    @ImageHelper(by_frame=True)
    def expand_to_cytoring(self,
                           labels: Mask,
                           image: Image = None,
                           distance: float = 1,
                           margin: int = 0,
                           thres: float = None,
                           relative: bool = True,
                           mask: Mask = None,
                           ) -> Mask:
        """
        Expands labels in the given mask by a fixed
        distance.

        :param labels:
        :param distance:
        :param margin:
        :param thres:
        :param relative:
        :param mask: If given, only expands labels into indices
            that are True in mask.

        :return:
        """
        # Expand initial seeds before applying expansion
        if margin:
            labels = segm.expand_labels(labels, margin)

        out = segm.expand_labels(labels, distance)
        out -= labels

        # Make threshold mask if needed
        if thres is not None:
            assert image is not None, 'No intensity image provided'
            if relative:
                # Use 98th percentile instead of max to avoid outliers
                thres *= np.percentile(image, 98)

            thres_mask = image >= thres
            if mask is not None:
                mask *= thres_mask
            else:
                mask = thres_mask

        # if mask is not None:
            out[~mask.astype(bool)] = 0

        return out

    @ImageHelper(by_frame=True)
    def remove_nuc_from_cyto(self,
                             cyto_mask: Mask,
                             nuc_mask: Mask,
                             val_match: bool = False,
                             erosion: bool = False
                             ) -> Mask:
        """
        Removes nuclei from a cytoplasmic mask, optionally
        eroding the final mask.

        :param cyto_mask:
        :param nuc_mask:
        :param val_match:
        :param erosion:

        :return:
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
                          method: str = 'ndi',
                          **kwargs
                          ) -> Mask:
        """
        Fills holes in a binary image. Two algorithms are available, one
        from SimpleITK and one from scipy. Neither algorithm fills holes
        on the border of the image (due to ambiguity in defining a hole).
        This function can optionally fill holes on the border by guessing
        which holes are holes in a larger object.

        :param mask:
        :param fill_border:
        :param iterations:
        :parma kernel_radius:
        :param max_length:
        :param in_place:
        :param method:
        :param kwargs:

        :return:

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
        Returns a binary level set of various shapes.

        :param image:
        :param levelset:
        :param size:
        :param center:
        :param label:

        :return:

        TODO:
            - size refers to square_size for checkerboard or radius for disk
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
        """Propagates an initial level set to edges found in an edge potential image.
        This function will likely not work well on an unmodified input image. Use the
        edge detection functions in ``Process`` to create an edge potential image.

        :param edge_potential:
        :param initial_level_set:
        :param curvature:
        :param propagation:
        :param iterations:

        :return:
        """
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
        """Level set segmentation method based on intensity values.

        :param image:
        :param initial_level_set:
        :param lower_thres:
        :param upper_thres:
        :param propagation:
        :param curvature:
        :param iterations:
        :param max_rms_error:

        :return:
        """
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
        """Computes the distance from the seeds based on a fast
        marching algorithm.

        :param image:
        :param seeds:
        :param n_points:

        :return:
        """
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
        """Runs morphological watershed from given seeds.

        :param image:
        :param mask:
        :param watershed_line:
        :param remove_large:

        :return:
        """
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
        """Runs morphological watershed on the image.

        :param image:
        :param watershed_line:

        :return:
        """
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
        """Applies a voting algorithm to determine the value
        of each pixel. Can be used to get the combined result
        from multiple masks.

        :param mask:
        :param label_undecided:

        :return:
        """
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

        TODO:
            - This should just call the Process version __wrapped__
            - Add citations
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

        if batch is None or batch >= image.shape[0]:
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
                      model_path: str = None,
                      weight_path: str = None,
                      batch: int = None,
                      ) -> Mask:
        """Wrapper for implementation of Misic. Original paper from
        `Panigrahi and colleagues`_.

        .. _Panigrahi and colleagues: https://elifesciences.org/articles/65151

        :param image:
        :param model_path:
        :param weight_path:
        :param batch:

        :return:

        TODO:
            - Make custom structure to speed up calculations
        """
        # Only import tensorflow if needed
        from celltk.utils.unet_model import MisicModel
        if model_path is None:
            # Original MiSiC model should have been installed
            model_path = pkg.resource_filename('celltk',
                                               'external/misic/MiSiCv2.h5')
        self.model = MisicModel(model_path)

        # Use model for predictions
        if batch is None or batch >= image.shape[0]:
            output = self.model.predict(image[:, :, :], )
        else:
            arrs = np.array_split(image, image.shape[0] // batch, axis=0)
            output = np.concatenate([self.model.predict(a) for a in arrs],
                                    axis=0)

        return output
