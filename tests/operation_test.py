import os
import sys

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import numpy as np
import pytest

from celltk.utils._types import Image, Mask
from celltk.utils.utils import ImageHelper
from celltk.core.operation import Operation

class UnitFunctions(Operation):
    def _generate_default_arrays(self):
        # Generate images - unique integers, type float
        img0 = np.zeros((5, 10, 10), dtype=np.float32)
        img0[:] = 0.0

        img1 = np.zeros((5, 10, 10), dtype=np.float32)
        img1[:] = 1.0

        # Masks - unique integers, type np.uint
        msk0 = np.zeros((5, 10, 10), dtype=np.uint16)
        msk0[:] = 0

        msk1 = np.zeros((5, 10, 10), dtype=np.uint16)
        msk1[:] = 1

        # Tracks - unique negative integers,

        trk0 = np.zeros((5, 10, 10), dtype=np.int16)
        trk0[:] = -1

        return [img0, img1], [msk0, msk1], [trk0]

    @ImageHelper(by_frame=False, as_tuple=False)
    def _array_single_stack(self,
                            image: Image,
                            mask: Mask
                            ) -> Image:
        """
        Expects img1 and msk1
        """
        assert image.shape == (5, 10, 10)
        assert mask.shape == (5, 10, 10)
        assert (image == 1.0).all()
        assert (mask == 1).all()

        return image

    @ImageHelper(by_frame=True, as_tuple=False)
    def _array_single_frames(self,
                             image: Image,
                             mask: Mask
                             ) -> Image:
        """
        Expects frames from img1 and msk1
        """
        assert image.shape == (10, 10)
        assert mask.shape == (10, 10)
        assert (image == 1.0).all()
        assert (mask == 1).all()

        return image

    @ImageHelper(by_frame=False, as_tuple=True)
    def _array_single_tuple(self,
                            image: Image,
                            mask: Mask
                            ) -> Image:
        """
        Expects all images and all masks in tuple
        """
        assert isinstance(image, tuple) and isinstance(mask, tuple)
        assert all(i.shape == (5, 10, 10) for i in image)
        assert all(m.shape == (5, 10, 10) for m in mask)
        assert (image[0] == 0.0).all()
        assert (image[1] == 1.0).all()
        assert (mask[0] == 0).all()
        assert (mask[1] == 1).all()

        return image


class TestOperation:
    """
    - Test __call__ method
    - Assure correct inputs are passed around
    """
    def test_passing_args(self):
        op = UnitFunctions()
        img, msk, trk = op._generate_default_arrays()

        # Test passing single stacks around
        op.add_function_to_operation('_array_single_stack')
        op.add_function_to_operation('_array_single_frames')
        op.add_function_to_operation('_array_single_tuple')
        op(img, msk, trk)

        # Reset the instance for more testing
        op = UnitFunctions()


