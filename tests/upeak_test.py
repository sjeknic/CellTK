import sys
import os

import numpy as np
import pytest

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import cellst
import cellst.utils.unet_model
from cellst.utils.upeak.peak_utils import segment_peaks_agglomeration
from cellst.utils.plot_utils import plot_trace_predictions

class TestUPeak():
    weight_path = "cellst/config/upeak_example_weights.tf"
    data_path = os.path.join(par, 'examples/example_traces.npy')
    def _upeak_model_creation(self):
        upeak = cellst.utils.unet_model.UPeakModel(self.weight_path)
        return upeak

    def test_upeak_model_creation(self):
        # Get a UPeakModel object
        upeak = self._upeak_model_creation()
        # No model should exist until data are passed
        with pytest.raises(AttributeError):
            upeak.model

        # Import example data
        #arr = np.load(os.path.abspath(self.data_path))
        exp = cellst.ExperimentArray.load(self.data_path)
        for k, arr in exp.items():
            arr = arr['nuc', 'fitc', 'median_intensity']
            out = upeak.predict(arr, roi=(0, 1, 2))

            plot_trace_predictions(arr, out[..., 1:])


class TestPeakSegmentation():
    arr_path = os.path.join(par, 'examples/example_traces.npy')
    pred_path = os.path.join(par, 'examples/example_predictions.npy')

    def __init__(self) -> None:
        self.test_load_traces()
        self.test_peak_segmentation()

    def test_load_traces(self):
        self.arr = np.load(self.arr_path)
        self.pred = np.load(self.pred_path)

        assert self.arr.shape[:2] == self.pred.shape[:2]

    def test_peak_segmentation(self):
        out = segment_peaks_agglomeration(self.arr, self.pred)


if __name__ == '__main__':
    #TestUPeak().test_upeak_model_creation()
    TestPeakSegmentation()