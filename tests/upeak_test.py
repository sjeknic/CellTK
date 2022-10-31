import sys
import os

import numpy as np
import pytest

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import celltk
import celltk.utils.unet_model
from celltk.utils.peak_utils import segment_peaks_agglomeration

class TestUPeak():
    weight_path = "config/upeak_example_weights.tf"
    data_path = "examples/example_traces.npy"

    def _upeak_model_creation(self):
        upeak = celltk.utils.unet_model.UPeakModel(self.weight_path)
        return upeak

    def test_upeak_model_creation(self):
        # Get a UPeakModel object
        upeak = self._upeak_model_creation()
        # No model should exist until data are passed
        with pytest.raises(AttributeError):
            upeak.model

        # Import example data
        arr = np.load(os.path.abspath(self.data_path))

        # Predict and plot
        upeak.predict(arr, roi=(1))


class TestPeakSegmentation():
    arr_path = "examples/example_traces.npy"
    pred_path = "examples/example_predictions.npy"

    def _run(self) -> None:
        self.test_peak_segmentation()

    def test_peak_segmentation(self):
        self.arr = np.load(self.arr_path)
        self.pred = np.load(self.pred_path)

        assert self.arr.shape[:2] == self.pred.shape[:2]

        # Only pass the peak value
        out = segment_peaks_agglomeration(self.arr, self.pred)

        assert out.shape == self.arr.shape
