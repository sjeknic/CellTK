import sys
import os

import numpy as np
import plotly.graph_objects as go

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import celltk
from celltk.utils.plot_utils import PlotHelper


class TestPlots:
    _exp_array_path = os.path.join(par, 'examples', 'example_experiment.hdf5')
    _keys = ['control_20K', 'ifny_10_20K']
    _show = False

    def test_each_plot(self) -> None:
        """"""
        # Load array and get np array data
        self.ph = PlotHelper()
        arr = celltk.ExperimentArray.load(self._exp_array_path)

        # Lists of 2D arrays for some plots
        data = arr[self._keys]['nuc', 'fitc', 'median_intensity']
        data2 = arr[self._keys]['nuc', 'cfp', 'median_intensity']

        # Lists of 1D arrays for the rest
        datum = [d[:, 0] for d in data]
        datum2 = [d[:, 0] for d in data2]

        # Go through the list of plots and generate each one
        # Randomly testing some options throughout
        bar = self.ph.bar_plot(datum, estimator=np.nanmean)
        if self._show: bar.show()
        assert isinstance(bar, go.Figure)

        contour = self.ph.contour2d_plot(datum[0], datum2[0], robust_z=True)
        if self._show: contour.show()
        assert isinstance(contour, go.Figure)

        heatmap = self.ph.heatmap2d_plot(datum[0], datum2[0], widget=True)
        if self._show: heatmap.show()
        assert isinstance(heatmap, go.FigureWidget)

        heatmap = self.ph.heatmap_plot(data[0], robust_z=True)
        if self._show: heatmap.show()
        assert isinstance(heatmap, go.Figure)

        histogram = self.ph.histogram_plot(datum, histnorm='probability')
        if self._show: histogram.show()
        assert isinstance(histogram, go.Figure)

        line = self.ph.line_plot(data, estimator=np.nanmean,
                                 err_estimator=np.nanstd)
        if self._show: line.show()
        assert isinstance(line, go.Figure)

        ridge = self.ph.ridgeline_plot(datum)
        if self._show: ridge.show()
        assert isinstance(ridge, go.Figure)

        scatter = self.ph.scatter_plot(datum, datum2)
        if self._show: scatter.show()
        assert isinstance(scatter, go.Figure)

    def test_trace_color(self) -> None:
        # Load array and get np array data
        self.ph = PlotHelper()
        arr = celltk.ExperimentArray.load(self._exp_array_path)

        data = arr['control_20K']['nuc', 'fitc', 'median_intensity']
        color = arr['control_20K']['nuc', 'fitc', 'peak_prob']
        thres = 0.5
        colors = ['black', 'red']

        gen = self.ph.trace_color_plot(data, color, thres, colors)
        for f in gen:
            if self._show: f.show()
            assert isinstance(f, go.Figure)
