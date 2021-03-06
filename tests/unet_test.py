import sys
import os
from glob import glob

import numpy as np
import imageio as iio
import pytest

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import celltk
import celltk.utils.unet_model

class TestUNet():
    weight_path = os.path.join(par, 'config/unet_example_cell_weights.tf')
    misic_path = os.path.join(par, 'celltk/external/misic/MiSiCv2.h5')
    # TODO: Add bacterial test images
    data_path = os.path.join(par, 'examples/live_cell_example')

    def _unet_model_creation(self, dims):
        unet = celltk.utils.unet_model.FluorUNetModel(dims, self.weight_path)
        return unet

    def test_unet_model_creation(self):
        # Load the data
        arrs = []
        for im in glob(os.path.join(self.data_path, '*l000*tif')):
            arrs.append(iio.imread(os.path.join(self.data_path, im)))
        img = np.stack(arrs, axis=0)

        # Get a UNetModel object
        dims = (img.shape[1], img.shape[2], 1)  # 1 input class
        unet = self._unet_model_creation(dims)

        # Import example data
        batch = 2
        arrs = np.array_split(img, img.shape[0] // batch, axis=0)
        output = np.concatenate([unet.predict(a, roi=2)
                                 for a in arrs], axis=0)

    def test_misic_model(self):
        self.model = celltk.utils.unet_model.MisicModel(self.misic_path)
        arrs = []
        for im in glob(os.path.join(self.data_path, '*l000*tif')):
            arrs.append(iio.imread(os.path.join(self.data_path, im)))
        img = np.stack(arrs, axis=0)
        output = self.model.predict(img[:, :, :], )


if __name__ == '__main__':
    TestUNet().test_unet_model_creation()
