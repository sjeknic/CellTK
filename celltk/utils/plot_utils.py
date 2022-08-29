import itertools
import functools
import warnings
from copy import deepcopy
from typing import Collection, Union, Callable, Generator, Tuple

import numpy as np
import sklearn.base as base
import sklearn.metrics as metrics
import sklearn.preprocessing as preproc
import scipy.stats as stats
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pcol
import plotly.subplots as psubplt
import colorcet as cc

import celltk.utils.estimator_utils


class PlotHelper:
    """
    Helper class for making plots. Most functions are simple wrappers around
    Plotly functions.
    """
    # Establishing the default format
    _template = 'simple_white'
    _font_families = "Graphik,Roboto,Helvetica Neue,Helvetica,Arial"
    _ax_col = '#444'
    _black = '#000000'
    _default_axis_layout = {
        'layer': 'above traces',
        'linewidth': 3,
        'linecolor': _ax_col,
        'showline': True,
        'tickfont': dict(color=_ax_col, size=18, family=_font_families),
        'ticklen': 7,
        'ticks': 'outside',
        'tickwidth': 2,
        'title': dict(font=dict(color='#242424', size=24, family=_font_families)),
        'zeroline': False,
    }
    _no_line_axis = _default_axis_layout.copy()
    _no_line_axis['showline'] = False
    _default_legend_layout = {
        'borderwidth': 0,
        'font': dict(color=_ax_col, size=18, family=_font_families),
        'itemsizing': 'constant',
        'itemwidth': 30,  # must be >= 30
        'title': {'font': dict(color=_ax_col, size=24,
                               family=_font_families)},
        'tracegroupgap': 10,  # must be >= 0
    }
    _default_error_bar_layout = {
        'color': _ax_col,
        'thickness': 2,
        'visible': True,
        'width': 0
    }

    # kwarg keys
    _line_kwargs = ('color', 'dash', 'shape', 'simplify', 'smoothing', 'width',
                    'hoverinfo')
    _marker_kwargs = ('color', 'line', 'opacity', 'size', 'symbol')
    _violin_kwargs = ('bandwidth', 'fillcolor', 'hoverinfo', 'jitter', 'line',
                      'marker', 'opacity', 'pointpos', 'points', 'span',
                      'width', 'hoverinfo')
    _bar_kwargs = ('hoverinfo', 'marker', 'width')

    class _Normalizer:
        def __init__(self, fitter: base.BaseEstimator) -> None:
            self._fitter = fitter()
            self._fitter_type = self._fitter.__str__().split('(')[0]
            self._fit = False

        def __call__(self, array: np.ndarray, scale_only: bool = False):
            """Fits the scaler on the first call and only applies
            on all subsequent calls."""
            if array is None:
                return
            elif self._fit:
                if scale_only:
                    if self._fitter_type in ('MinMaxScaler'):
                        return array * self._fitter.scale_
                    else:
                        return array / self._fitter.scale_
                else:
                    return self._fitter.transform(array)
            else:
                self._fit = True
                return self._fitter.fit_transform(array)

    def _build_colormap(self,
                        colors: Union[str, Collection[str]],
                        number: int,
                        alpha: float = 1.0
                        ) -> Generator:
        """Returns an infinite color generator for an arbitrary
        colormap or colorscale."""
        if colors is None:
            _col = cc.glasbey_dark
        elif isinstance(colors, (list, tuple, np.ndarray)):
            _col = colors
        else:
            # Check in a few places
            try:
                _col = sns.color_palette(colors, number)
            except ValueError:
                try:
                    # Try getting it from Plotly
                    _col = pcol.get_colorscale(colors)
                    _col = pcol.colorscale_to_colors(_col)
                except Exception:  # This will be PlotlyError, find it
                    raise ValueError(f'Could not get colorscale for {colors}')

        _col = [self._format_colors(c, alpha) for c in _col]
        return itertools.cycle(_col)

    def _build_symbolmap(self,
                         symbols: Union[str, Collection[str]]
                         ) -> Generator:
        """
        TODO:
            - Add check that symbols are valid
        """
        if isinstance(symbols, (str, int, float, type(None))):
            symbols = [symbols]

        return itertools.cycle(symbols)

    def _apply_format_figure(self,
                             figure: go.Figure,
                             figsize: Tuple[int] = (None, None),
                             title: str = None,
                             x_label: str = None,
                             y_label: str = None,
                             x_limit: Tuple[float] = None,
                             y_limit: Tuple[float] = None,
                             legend: bool = None,
                             tick_size: float = None,
                             axis_label_size: float = None,
                             axis_type: str = 'default',
                             margin: str = 'auto',
                             row: int = None,
                             col: int = None,
                             **kwargs
                             ) -> go.Figure:
        """
        TODO:
            - Standardize keyword inputs and simplify. e.g. ?_label for x_label
              and y_label. And input to each function as part of kwargs.
        """
        # Default layouts
        if axis_type == 'default':
            x_axis_layout = deepcopy(self._default_axis_layout)
            y_axis_layout = deepcopy(self._default_axis_layout)
        elif axis_type == 'noline':
            x_axis_layout = deepcopy(self._no_line_axis)
            y_axis_layout = deepcopy(self._no_line_axis)
        else:
            raise ValueError(f'Did not understand axis type {axis_type}.')

        figure_layout = {'template': self._template}

        # Apply changes to margins
        if isinstance(margin, (float, int)):
            m = int(margin)
            figure_layout.update({'margin': dict(l=m, r=m, b=m, t=m)})
        elif margin in (False, 'zero', 'tight'):
            figure_layout.update({'margin': dict(l=0, r=0, t=0, b=0)})

        # Updates only made if not None, preserves old values
        if legend is not None:
            figure_layout.update({'showlegend': legend})
        if x_label is not None:
            x_axis_layout['title'].update({'text': x_label})
        if y_label is not None:
            y_axis_layout['title'].update({'text': y_label})
        if x_limit is not None:
            x_axis_layout.update({'range': x_limit})
        if y_limit is not None:
            y_axis_layout.update({'range': y_limit})
        if title is not None:
            figure_layout.update({'title': title})
        if figsize[0] is not None:
            figure_layout.update({'height': figsize[0]})
        if figsize[1] is not None:
            figure_layout.update({'width': figsize[1]})
        if tick_size is not None:
            x_axis_layout['tickfont'].update({'size': tick_size})
            y_axis_layout['tickfont'].update({'size': tick_size})
        if axis_label_size is not None:
            x_axis_layout['title']['font'].update({'size': axis_label_size})
            y_axis_layout['title']['font'].update({'size': axis_label_size})

        # Apply changes
        figure.update_layout(**figure_layout, **kwargs)
        figure.update_xaxes(**x_axis_layout, row=row, col=col)
        figure.update_yaxes(**y_axis_layout, row=row, col=col)

        return figure

    @staticmethod
    def _format_colors(color: str, alpha: float = None) -> str:
        """Converst hexcode colors to RGBA to allow transparency"""
        def _convert_to_255(values):
            # Otherwise, it's fine for 0-255
            if len(values) == 3:
                alpha = None
            elif len(values) == 4:
                alpha = values[-1]

            if any(r > 0 and r < 1 for r in values[:3]):
                # This means at least one value is 0-1
                values = [int(round(r * 255)) for r in values[:3]]

                # For some odd reason, It seems som colors don't
                # render properly with values exactly 1.
                # Don't ask me why. I'm just making them 2 for now
                values = [v if v != 1 else 2 for v in values]

                if alpha is not None:
                    values += [alpha]
            return values

        if isinstance(color, (list, tuple)):
            if all([isinstance(f, (float, int)) for f in color]):
                # Assume rgb
                if alpha:
                    color += (alpha, )
                else:
                    color += (1.,)
                color = _convert_to_255(color)
                color_str = str(tuple([c for c in color]))
            else:
                # Assume first value is alpha
                alpha = alpha if alpha else color[0]
                if alpha < 0.125: alpha = 0.125
                values = pcol.unlabel_rgb(color[1]) + (alpha,)
                values = _convert_to_255(values)
                color_str = str(tuple([c for c in values]))

            return f'rgba{color_str}'
        elif isinstance(color, str):
            # Convert hex to rgba
            if color[0] == '#':
                # Check if alpha channel exists
                if len(color.split('#')[-1]) == 8:
                    # overwrites existing alpha
                    # convert hexademical to int, divide by max
                    alpha = int(color[1:3], 16) / 255.
                    color = pcol.hex_to_rgb('#' + color[3:])
                else:
                    color = pcol.hex_to_rgb(color)

                if alpha:
                    color += (alpha,)
                color = _convert_to_255(color)
                color_str = str(tuple([c for c in color]))
                if len(color) == 4:
                    return f'rgba{color_str}'
                else:
                    return f'rgb{color_str}'
            elif color[:3] in ('rgb'):
                vals = pcol.unlabel_rgb(color)
                vals = _convert_to_255(vals)
                if alpha:
                    vals += (alpha, )
                color_str = str(tuple([v for v in vals]))
                return f'rgba{color_str}'
            else:
                try:
                    vals = mcolors.to_rgba(color)
                    vals = _convert_to_255(vals)
                    if alpha:
                        vals = (*vals[:-1], alpha)
                    color_str = str(tuple([v for v in vals]))
                    return f'rgba{color_str}'
                except ValueError:
                    raise ValueError(f'Did not understand color {color}')

    @staticmethod
    def _format_arrays(arrays: Union[np.ndarray, Collection[np.ndarray]]
                       ) -> Collection[np.ndarray]:
        """"""
        _t = (np.integer, np.float, np.ndarray, int, float)
        if isinstance(arrays, np.ndarray):
            # If the input is an array, put it in a list
            # Cannot be object array, cannot pass array of arrays
            assert arrays.dtype != 'object', 'Cannot use object arrays'
            out = [arrays]
        elif isinstance(arrays, _t):
            out = [np.array(arrays)]
        elif arrays is None or arrays == []:
            # Empty input is just returned
            return arrays
        else:
            # Basic check is that everything is an array or numeric type
            if not all(isinstance(a, _t) for a in arrays):
                raise TypeError('Received non-numeric type')

            # Some plotly functions only take arrays, cast everything to array
            out = [np.array(a) for a in arrays]

        return out

    @staticmethod
    def _format_keys(keys: Collection[str],
                     default: str = 'trace',
                     add_cell_numbers: bool = True,
                     arrays: Collection[np.ndarray] = []
                     ) -> Collection[str]:
        if not isinstance(keys, (list, tuple, np.ndarray)):
            keys = [keys]

        # Make sure everything is a string
        keys = [str(k) for k in keys]
        # Extend the list if not enough keys
        if len(keys) < len(arrays):
            needed = len(arrays) - len(keys)
            old_len = len(keys)
            for n in range(needed):
                keys.append(f'{default}_{n + old_len}')

        # Add number of cells
        if add_cell_numbers:
            num_cells = [a.shape[0] if len(a.shape) >= 1 else 1
                         for a in arrays]
            keys = [f'{k} | n={n}' for k, n in zip(keys, num_cells)]

        return keys

    def _build_estimator_func(self,
                              func: Union[Callable, str, functools.partial],
                              *args, **kwargs
                              ) -> functools.partial:
        """"""
        if isinstance(func, functools.partial):
            # Assume user already made the estimator
            return func
        elif isinstance(func, Callable):
            return functools.partial(np.apply_along_axis, func, 0)
        elif isinstance(func, str):
            try:
                func = getattr(celltk.utils.estimator_utils, func)
                return functools.partial(func, *args, **kwargs)
            except AttributeError:
                try:
                    func = getattr(np, func)
                    return functools.partial(func, axis=0, *args, **kwargs)
                except AttributeError:
                    raise ValueError(f'Did not understand estimator {func}')

    def _build_normalizer_func(self,
                               func: Union[str, Callable]
                               ) -> functools.partial:
        """"""
        if isinstance(func, Callable):
            return func
        elif isinstance(func, str):
            if func == 'minmax': fitter = 'MinMax'
            elif func == 'maxabs': fitter = 'MaxAbs'
            elif func == 'standard': fitter = 'Standard'
            elif func == 'robust': fitter = 'Robust'
            else: raise ValueError(f'Did not understand normalizer {func}.')
            fitter += 'Scaler'

            fitter = getattr(preproc, fitter)
            return self._Normalizer(fitter)

    def line_plot(self,
                  arrays: Collection[np.ndarray],
                  keys: Collection[str] = [],
                  estimator: Union[Callable, str, functools.partial] = None,
                  err_estimator: Union[Callable, str, functools.partial] = None,
                  normalizer: Union[Callable, str] = None,
                  colors: Union[str, Collection[str]] = None,
                  alpha: float = 1.0,
                  time: Union[Collection[np.ndarray], np.ndarray] = None,
                  add_cell_numbers: bool = True,
                  legend: bool = True,
                  figure: Union[go.Figure, go.FigureWidget] = None,
                  figsize: Tuple[int] = (None, None),
                  margin: str = 'auto',
                  title: str = None,
                  x_label: str = None,
                  y_label: str = None,
                  x_limit: Tuple[float] = None,
                  y_limit: Tuple[float] = None,
                  tick_size: float = None,
                  axis_label_size: float = None,
                  widget: bool = False,
                  gl: bool = False,
                  row: int = None,
                  col: int = None,
                  **kwargs
                  ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly Figure object plotting lines of the given arrays. Each
        array is interpreted as a separate condition and is plotted in a
        different color.

        :param arrays: List of arrays to plot. Assumed structure is n_cells x
            n_features. Arrays must be two-dimensional, so if only one sample,
            use np.newaxis or np.expand_dims.
        :param keys: Names corresponding to the data arrays. If not provided,
            the keys will be integers.
        :param estimator: Function for aggregating observations from multiple
            cells. For example, if estimator is np.mean, the mean of all of the
            cells will be plotted instead of a trace for each cell. Can be
            a function, name of numpy function, name of function in
            estimator_utils, or a functools.partial object. If a function or
            functools.partial object, should input a 2D array and return a
            1D array.
        :param err_estimator: Function for estimating error bars from multiple
            cells. Can be
            a function, name of numpy function, name of function in
            estimator_utils, or a functools.partial object. If a function or
            functools.partial object, should input a 2D array and return a
            1D or 2D array. If output is 1D, errors will be symmetric
            If output is 2D, the first dimension is used for the high
            error and second dimension is used for the low error.
        :param normalizer: If given, used to normalize the data after applying
            the estimators. Normalizes the error as well. Can be 'minmax',
            'maxabs', 'standard', 'robust', or a callable that inputs an array
            and outputs an array of the same shape.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey.
        :param alpha: Opacity of the line colors.
        :param time: Time axis for the plot. Must be the same size as the
            second dimension of arrays.
        :param add_cell_numbers: If True, adds the number of cells to each key
            in keys.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param gl: If True, switches to using a WebGL backend. Much faster for
            large datasets, but some features may not be available. May not
            work in all contexts.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Depending on name, passed to the "line" keyword
            argument of go.Scatter or as keyword arguments for go.Scatter.
            The following kwargs are passed to "line": 'color', 'dash',
            'shape', 'simplify', 'smoothing', 'width', 'hoverinfo'

        :return: Figure object

        :raises AssertionError: If any item in arrays is not two dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        :raises TypeError: If time is not an np.ndarray or collection of
            np.ndarray.

        TODO:
            - Add more generalized line_plot, turn this to time_plot
        """
        # Format inputs
        arrays = self._format_arrays(arrays)
        assert all([a.ndim == 2 for a in arrays])
        keys = self._format_keys(keys, default='line', arrays=arrays,
                                 add_cell_numbers=add_cell_numbers)
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays), alpha)
        if normalizer: normalizer = self._build_normalizer_func(normalizer)
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        line_kwargs = {k: v for k, v in kwargs.items()
                       if k in self._line_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._line_kwargs}

        # Build the figure and start plotting
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        # Switch to using WebGL if requested
        if gl:
            scatter = go.Scattergl
        else:
            scatter = go.Scatter

        for idx, (arr, key) in enumerate(zip(arrays, keys)):
            # err_estimator is used to set the bounds for the shaded region
            if err_estimator:
                err_arr = err_estimator(arr)
            else:
                err_arr = None

            # estimator is used to condense all the lines to a single line
            if estimator:
                arr = estimator(arr)
                if arr.ndim == 1: arr = arr[None, :]

            # normalizer can be used to set the range of arr and err_arr
            if normalizer:
                arr = normalizer(arr.reshape(-1, 1)).reshape(arr.shape)
                if err_arr is not None:
                    err_arr = normalizer(
                        err_arr.reshape(-1, 1), scale_only=True
                    ).reshape(err_arr.shape)

            if time is None:
                x = np.arange(arr.shape[1])
            elif isinstance(time, (tuple, list)):
                x = time[idx]
            elif isinstance(time, np.ndarray):
                x = time
            else:
                raise TypeError('Did not understand time of '
                                f'type {type(time)}.')

            lines = []
            _legend = True
            for a, y in enumerate(arr):
                if a: _legend = False
                line_kwargs.update({'color': next(colors)})

                lines.append(
                    scatter(x=x, y=y, legendgroup=key, name=key,
                            showlegend=_legend, mode='lines',
                            line=line_kwargs, **kwargs)
                )

                if err_arr is not None:
                    if err_arr.ndim == 1:
                        # Assume it's high and low deviation from y
                        hi = np.nansum([y, err_arr], axis=0)
                        lo = np.nansum([y, -err_arr], axis=0)
                    else:
                        lo = err_arr[0, :]
                        hi = err_arr[-1, :]

                    lines.append(
                        scatter(x=np.hstack([x, x[::-1]]),
                                y=np.hstack([hi, lo[::-1]]), fill='tozerox',
                                fillcolor=self._format_colors(line_kwargs['color'], 0.25),
                                showlegend=False, legendgroup=key,
                                name=key, line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo='skip')
                    )

            fig.add_traces(lines, rows=row, cols=col)

        # Upate the axes and figure layout
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        legend, tick_size, axis_label_size,
                                        axis_type='default', margin=margin,
                                        row=row, col=col)

        return fig

    def scatter_plot(self,
                     x_arrays: Collection[np.ndarray] = [],
                     y_arrays: Collection[np.ndarray] = [],
                     keys: Collection[str] = [],
                     estimator: Union[Callable, str, functools.partial] = None,
                     err_estimator: Union[Callable, str, functools.partial] = None,
                     normalizer: Union[Callable, str] = None,
                     scatter_mode: str = 'markers',
                     colors: Union[str, Collection[str]] = None,
                     alpha: float = 1.0,
                     symbols: Union[str, Collection[str]] = None,
                     add_cell_numbers: bool = True,
                     legend: bool = True,
                     figure: Union[go.Figure, go.FigureWidget] = None,
                     figsize: Tuple[int] = (None, None),
                     margin: str = 'auto',
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     x_limit: Tuple[float] = None,
                     y_limit: Tuple[float] = None,
                     tick_size: float = None,
                     axis_label_size: float = None,
                     widget: bool = False,
                     gl: bool = False,
                     row: int = None,
                     col: int = None,
                     **kwargs
                     ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly Figure object containing a scatter plot of the given
        arrays. Each array is interpreted as a separate condition and is
        plotted in a different color or with a different marker.

        :param x_arrays: List of arrays that set the x-coordinates to plot.
            Each array is assumed to be a different condition.
        :param y_arrays: List of arrays that set the y-coordinates to plot.
            Each array is assumed to be a different condition.
        :param keys: Names corresponding to the data arrays. If not provided,
            the keys will be integers.
        :param estimator: Function for aggregating observations from multiple
            cells. For example, if estimator is np.mean, the mean of all of the
            cells will be plotted instead of a point for each cell. Can be
            a function, name of numpy function, name of function in
            estimator_utils, or a functools.partial object. If a function or
            functools.partial object, should input a 2D array and return a
            1D array.
        :param err_estimator: Function for estimating vertical error bars.
            Can be a function, name of numpy function, name of
            function in estimator_utils, or a functools.partial object. If a
            function or functools.partial object, should input a 2D array and
            return a 1D or 2D array. If output is 1D, errors will be symmetric.
            If output is 2D, the first dimension is used for the high
            error and second dimension is used for the low error. Only
            applies to the y-dimension. Horizontal error bars not currrently
            implemented.
        :param normalizer: If given, used to normalize the data after applying
            the estimators. Normalizes the error as well. Can be 'minmax' or
            'maxabs', or a callable that inputs an array and outputs an array
            of the same shape.
        :param scatter_mode: Drawing mode for the traces. Can be 'markers',
            'lines', or 'lines+markers'.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param alpha: Opacity of the marker fill colors.
        :param add_cell_numbers: If True, adds the number of cells to each key
            in keys.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param gl: If True, switches to using a WebGL backend. Much faster for
            large datasets, but some features may not be available. May not
            work in all contexts.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Depending on name, passed to the "marker" keyword
            argument of go.Scatter or as keyword arguments for go.Scatter.
            The following kwargs are passed to "marker": 'color', 'line',
            'opacity', 'size', 'symbol'.

        :return: Figure object

        :raises AssertionError: If both x_arrays and y_arrays are given,
            but have different lengths.
        :raises AssertionError: If not all items in either array are
            np.ndarray.
        :raises AssertionError: If not all items in arrays have the same
            number of columns.
        :raises AssertionError: If any item in arrays has more than 3 columns.
        :raises AssertionError: If figsize is not a tuple of length two.

        TODO:
            - Allow normalization of both x and y arrays
        """
        # Format inputs - cast to np.ndarray as needed
        x_arrays = self._format_arrays(x_arrays)
        y_arrays = self._format_arrays(y_arrays)
        if x_arrays and y_arrays: assert len(x_arrays) == len(y_arrays)
        if y_arrays:
            keys = self._format_keys(keys, default='trace', arrays=y_arrays,
                                     add_cell_numbers=add_cell_numbers)
        elif x_arrays:
            keys = self._format_keys(keys, default='trace', arrays=x_arrays,
                                     add_cell_numbers=add_cell_numbers)
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors,
                                      max(len(x_arrays), len(y_arrays)),
                                      alpha)
        symbols = self._build_symbolmap(symbols)
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        if normalizer: normalizer = self._build_normalizer_func(normalizer)
        marker_kwargs = {k: v for k, v in kwargs.items()
                         if k in self._marker_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._marker_kwargs}

        # Build the figure and start plotting
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        # Switch to WebGL backend
        if gl:
            scatter = go.Scattergl
        else:
            scatter = go.Scatter

        traces = []
        zipped = itertools.zip_longest(x_arrays, y_arrays, keys,
                                       fillvalue=None)
        for idx, (xarr, yarr, key) in enumerate(zipped):
            # err_estimator is used to set error bars
            if err_estimator:
                err_arr = err_estimator(yarr)
            else:
                err_arr = None

            # estimator is used to condense all the cells to a single point
            if estimator:
                yarr = estimator(yarr)

            # Normalizer is used to get data onto the same scale
            if normalizer:
                yarr = normalizer(yarr.reshape(-1, 1)).reshape(yarr.shape)
                if err_arr is not None:
                    # If err_arr is only 1 dim, scale only, otherwise shift too
                    scale_only = err_arr.ndim == 1
                    err_arr = normalizer(
                        err_arr.reshape(-1, 1), scale_only=scale_only
                    ).reshape(err_arr.shape)

            # Assign to x and y:
            x = np.squeeze(xarr) if xarr is not None else None
            y = np.squeeze(yarr) if yarr is not None else None

            # Calculate error bars for y axis
            error_x = None
            error_y = None
            if err_arr is not None:
                error_y = self._default_error_bar_layout.copy()
                error_y.update({'type': 'data'})
                if err_arr.ndim == 1:
                    # Assume symmetric
                    error_y.update({'array': err_arr, 'symmetric': True})
                elif err_arr.ndim == 2:
                    # If 2-dimensions, assume it represents high and low bounds
                    err_plus = err_arr[0, :] - y
                    err_minus = y - err_arr[-1, :]
                    error_y.update({'array': err_plus,
                                    'arrayminus': err_minus})

            marker_kwargs.update(dict(color=next(colors),
                                      symbol=next(symbols)))
            traces.append(
                scatter(x=x, y=y, legendgroup=key, name=key,
                        showlegend=legend, mode=scatter_mode,
                        error_x=error_x, error_y=error_y,
                        marker=marker_kwargs, **kwargs)
            )

        fig.add_traces(traces, rows=row, cols=col)

        # Apply formatting and return
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        legend, tick_size, axis_label_size,
                                        axis_type='default', margin=margin,
                                        row=row, col=col)

        return fig

    def bar_plot(self,
                 arrays: Collection[np.ndarray],
                 keys: Collection[str] = [],
                 estimator: Union[Callable, str, functools.partial] = None,
                 err_estimator: Union[Callable, str, functools.partial] = None,
                 ax_labels: Collection[str] = None,
                 colors: Union[str, Collection[str]] = None,
                 alpha: float = 1.0,
                 orientation: str = 'v',
                 barmode: str = 'group',
                 add_cell_numbers: bool = True,
                 legend: bool = True,
                 figure: Union[go.Figure, go.FigureWidget] = None,
                 figsize: Tuple[int] = (None, None),
                 margin: str = 'auto',
                 title: str = None,
                 x_label: str = None,
                 y_label: str = None,
                 x_limit: Tuple[float] = None,
                 y_limit: Tuple[float] = None,
                 tick_size: float = None,
                 axis_label_size: float = None,
                 widget: bool = False,
                 row: int = None,
                 col: int = None,
                 **kwargs
                 ) -> Union[go.Figure, go.FigureWidget]:
        """Builds a Plotly Figure object plotting bars from the given arrays. Each
        array is interpreted as a separate condition and is plotted in a
        different color.

        :param arrays: List of arrays to plot. Assumed structure is n_cells x
            n_features. Each feature is plotted as a separate bar group.
            Arrays must be two-dimensional, so if only one sample,
            use np.newaxis or np.expand_dims.
        :param keys: Names corresponding to the data arrays. If not provided,
            the keys will be integers.
        :param estimator: Function for aggregating observations from multiple
            cells. For example, if estimator is np.mean, the mean of all of the
            cells will be plotted instead of a bar for each cell. Can be
            a function, name of numpy function, name of function in
            estimator_utils, or a functools.partial object. If a function or
            functools.partial object, should input a 2D array and return a
            1D array.
        :param err_estimator: Function for estimating error bars. Can be
            a function, name of numpy function, name of function in
            estimator_utils, or a functools.partial object. If a function or
            functools.partial object, should input a 2D array and return a
            1D or 2D array. If output is 1D, errors will be symmetric
            If output is 2D, the first dimension is used for the high
            error and second dimension is used for the low error.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param alpha: Opacity of the marker fill colors.
        :param ax_labels: Labels for the categorical axis.
        :param orientation: Orientation of the bar plot.
        :param barmode: Keyword argument describing how to group the bars.
            Options are 'group', 'overlay', 'relative', and 'stack'.
        :param add_cell_numbers: If True, adds the number of cells to each key
            in keys.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is vertical.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Depending on name, passed to go.Bar or to
            go.Figure.update_traces(). The following kwargs are passed to
            go.Bar: 'hoverinfo', 'marker', 'width'.

        :return: Figure object

        :raises AssertionError: If not all items in arrays are np.ndarray.
        :raises AssertionError: If orientation is a disallowed value.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        # Format data
        arrays = self._format_arrays(arrays)
        keys = self._format_keys(keys, default='bar', arrays=arrays,
                                 add_cell_numbers=add_cell_numbers)
        assert orientation in ('v', 'h', 'horizontal', 'vertical')
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays), alpha)
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        bar_kwargs = {k: v for k, v in kwargs.items()
                      if k in self._bar_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._bar_kwargs}

        # Build the figure and start plotting
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        for idx, (arr, key) in enumerate(zip(arrays, keys)):
            # err_estimator is used to calculate errorbars
            if err_estimator:
                err_arr = err_estimator(arr)
            else:
                err_arr = None

            # estimator is used to condense all the data to a single point
            if estimator:
                arr = np.squeeze(estimator(arr))

            # Set the data key based on orientation
            error_x = None
            error_y = None
            if orientation in ('v', 'vertical'):
                y = arr
                x = ax_labels if ax_labels else None
                if err_arr is not None:
                    error_x = None
                    error_y = self._default_error_bar_layout.copy()
                    error_y.update({'type': 'data'})
                    if err_arr.ndim in (0, 1):
                        # Assume symmetric
                        error_y.update({'array': err_arr, 'symmetric': True})
                    elif err_arr.ndim == 2:
                        # Assume that they are already set based on the mean
                        # value, so that needs to be subtracted
                        err_plus = arr - err_arr[0, :]
                        err_minus = err_arr[-1, :] - arr
                        error_y.update({'array': err_plus,
                                        'arrayminus': err_minus})
            elif orientation in ('h', 'horizontal'):
                y = ax_labels if ax_labels else None
                x = arr
                if err_arr is not None:
                    error_y = None
                    error_x = self._default_error_bar_layout.copy()
                    error_x.update({'type': 'data'})
                    if err_arr.ndim in (0, 1):
                        # Assume symmetric
                        error_x.update({'array': err_arr, 'symmetric': True})
                    elif err_arr.ndim == 2:
                        # Assume that they are already set based on the mean
                        # value, so that needs to be subtracted
                        err_plus = arr - err_arr[0, :]
                        err_minus = err_arr[-1, :] - arr
                        error_x.update({'array': err_plus,
                                        'arrayminus': err_minus})

            # Set up the colors
            _c = next(colors)
            bar_kwargs.update({'marker_color': _c})

            trace = go.Bar(x=x, y=y, error_x=error_x, error_y=error_y,
                           name=key, **bar_kwargs)
            fig.add_traces(trace, rows=row, cols=col)

        # Format plot on the way out
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        legend, tick_size, axis_label_size,
                                        axis_type='default',
                                        barmode=barmode, margin=margin,
                                        row=row, col=col)
        fig.update_traces(**kwargs, row=row, col=col)

        return fig

    def histogram_plot(self,
                       arrays: Collection[np.ndarray],
                       keys: Collection[str] = [],
                       histfunc: str = 'count',
                       histnorm: str = "",
                       normalizer: Union[Callable, str] = None,
                       nbins: int = None,
                       binsize: float = None,
                       bargap: float = None,
                       bargroupgap: float = None,
                       cumulative: bool = False,
                       include_histogram: bool = True,
                       include_kde: bool = False,
                       bandwidth: Union[str, float] = None,
                       extend_kde: Union[float, bool] = 0,
                       fill_kde: Union[bool, str] = False,
                       colors: Union[str, Collection[str]] = None,
                       alpha: float = 1.0,
                       orientation: str = 'v',
                       barmode: str = 'group',
                       add_cell_numbers: bool = True,
                       legend: bool = True,
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       margin: str = 'auto',
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       tick_size: float = None,
                       axis_label_size: float = None,
                       widget: bool = False,
                       row: int = None,
                       col: int = None,
                       **kwargs
                       ) -> Union[go.Figure, go.FigureWidget]:
        """Builds a Plotly Figure object plotting a histogram of each of the
        given arrays. Each array is interpreted as a separate condition and
        is plotted in a different color.

        :param arrays: List of arrays to plot histograms.
        :param keys: Names corresponding to the data arrays. If not provided,
            the keys will be integers.
        :param histfunc: Specifies the binning function used for
            this histogram trace. If “count”, the histogram values
            are computed by counting the number of values lying
            inside each bin. Can also be “sum”, “avg”, “min”, “max”.
        :param histnorm: Specifies the type of normalization used
            for this histogram trace. If “”, the span of each bar
            corresponds to the number of occurrences  If “percent” /
            “probability”, the span of each bar corresponds to the
            percentage / fraction of occurrences with respect to
            the total number of sample points (here, the sum of
            all bin HEIGHTS equals 100% / 1). If “density”, the
            span of each bar corresponds to the number of
            occurrences in a bin divided by the size of the bin
            interval. If "probability density", the area of each
            bar corresponds to the probability that an event will
            fall into the corresponding bin (here, the sum of all
            bin AREAS equals 1).
        :param normalizer: If given, used to normalize the data after applying
            the estimators. Normalizes the error as well. Can be 'minmax' or
            'maxabs', or a callable that inputs an array and outputs an array
            of the same shape.
        :param nbins: Approximate number of bins to use. Ignored if
            binsize is set.
        :param binsize: Size of each bin.
        :param bargap: Gap between bars in adjacent locations.
        :param bargroupgap: Gap between bars in the same location.
        :param cumulative: If True, the histogram will plot cumulative
            occurances.
        :param include_histogram: If True, plot the histogram as a series
            of bars.
        :param include_kde: If True, calculate and plot a Gaussian kernel
            density estimate of the data. NOTE: Not currently normalized,
            so only plots the probability density function.
        :param bandwidth: Value of the bandwidth or name of method to
            estimate a good value of the bandwidth. Method options are
            'scott' and 'silverman'. If None, uses 'scott'.
        :param extend_kde: Boolean or number of bandwidth lengths to extend
            the plot of the kernel density estimate. False or value of 0 means
            the kernel density estimate will only be plotted for the values of
            the data provided. If True, the default value is 3.
        :param fill_kde: If True, fills in the area of the kernel density
            estimate. By default, fills to a value of 0 on the axis. Can
            also be a string to specify a different fill method. These options
            are from Plotly and include: 'tozerox', 'tozeroy', 'tonextx',
            'tonexty', 'tonext', 'toself'.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param alpha: Opacity of the fill color of the histogram as a float
            in the range [0, 1].
        :param orientation: Orientation of the bar plot.
        :param barmode: Keyword argument describing how to group the bars.
            Options are 'group', 'overlay', 'stack', and 'relative'.
        :param add_cell_numbers: If True, adds the number of cells to each key
            in keys.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is vertical.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Depending on name, kwargs are passed to either
            go.Histogram or the line kwarg of go.Scatter if a kernel density
            estimate is included. The following kwargs are passed to "line":
            'color', 'dash', 'shape', 'simplify', 'smoothing', 'width',
            'hoverinfo'.

        :return: Figure object

        :raises AssertionError: If not all items in arrays are np.ndarray.
        :raises AssertionError: If orientation is a disallowed value.
        :raises AssertionError: If figsize is not a tuple of length two.

        TODO:
            - Normalization of KDEs is slighlty imprecise. This is due to not
              having the exact bins from go.Histogram. So a new histogram is
              calculated with np.histogram. They, for whatever reason, don't
              line up exactly. In order to change this, the histogram should
              be calculated with np.histogram and the plot should be made
              using go.Bar. This is not high priority, as the bins are very
              close. There might even be a way to pre-calculate exactly the
              bins that go.Histogram will use, using np.linspace.
        """
        # Format data
        arrays = self._format_arrays(arrays)
        keys = self._format_keys(keys, default='dist', arrays=arrays,
                                 add_cell_numbers=add_cell_numbers)
        assert orientation in ('v', 'h', 'horizontal', 'vertical')
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays), alpha)
        if normalizer: normalizer = self._build_normalizer_func(normalizer)
        line_kwargs = {k: v for k, v in kwargs.items()
                       if k in self._line_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._line_kwargs}

        # Build the figure and start plotting
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        for idx, (arr, key) in enumerate(zip(arrays, keys)):
            if orientation in ('v', 'vertical'):
                y = None
                x = arr
            elif orientation in ('h', 'horizontal'):
                y = arr
                x = None

            if normalizer:
                if x is not None:
                    x = normalizer(x.reshape(-1, 1)).reshape(x.shape)
                if y is not None:
                    y = normalizer(y.reshape(-1, 1)).reshape(y.shape)

            # Set up the colors
            _c = next(colors)
            marker_kwargs = {'color': _c, 'line': {'color': _c}}
            line_kwargs.update({'color': _c})
            cum_kwargs = {'enabled': cumulative}

            # nbins refers to MAX number of bins, so to keep things
            # consistent for the kde, re-calculate the bins
            if nbins and not binsize:
                binsize = (np.nanmax(arr) - np.nanmin(arr)) / nbins

            # Make individual distributions on the plot
            if include_histogram:
                trace = go.Histogram(x=x, y=y, name=key, legendgroup=key,
                                     histfunc=histfunc, histnorm=histnorm,
                                     orientation=orientation,
                                     cumulative=cum_kwargs,
                                     xbins=dict(size=binsize),
                                     ybins=dict(size=binsize),
                                     marker=marker_kwargs,
                                     **kwargs)
                fig.add_traces(trace, rows=row, cols=col)

            # Estimate and plot KDE
            if include_kde:
                if histfunc != 'count':
                    warnings.warn('KDE not supported for histogram '
                                  f'function {histfunc}.')
                    break

                data = arr[~np.isnan(arr)]

                # Set up the KDE estimator
                kde = stats.gaussian_kde(data, bw_method=bandwidth)

                # Determine the support set to plot
                if extend_kde is True:
                    extend_kde = 3  # default value
                elif extend_kde is False:
                    extend_kde = 0
                left = data.min() - kde.factor * extend_kde
                right = data.max() + kde.factor * extend_kde
                if not binsize:
                    if include_histogram:
                        warnings.warn('Set the binsize or number of bins to'
                                      ' ensure consistent KDE estimation.')
                    binsize = (data.max() - data.min()) / 100

                num = np.ceil((data.max() - data.min()) / binsize).astype(int)
                support = np.linspace(left, right, num)

                # Calculate the estimated kernel density
                if cumulative:
                    kde_line = np.array([kde.integrate_box_1d(support[0], s)
                                         for s in support])
                else:
                    kde_line = kde.evaluate(support)

                # In order to scale kde, calculate new histogram
                if histnorm != 'probability density':
                    hist, edges = np.histogram(data, num, density=False)

                    if histnorm in ("probability",):
                        hist = hist.astype(float) / hist.sum()
                    elif histnorm in ("percent",):
                        hist = hist.astype(float) / hist.sum() * 100
                    elif histnorm in ("density",):
                        hist = hist.astype(float) / np.diff(edges)

                    if cumulative:
                        if histnorm in ("probability", "percent", ""):
                            hist = hist.cumsum()
                        elif histnorm in ("density",):
                            hist = (hist * np.diff(edges)).cumsum()

                        kde_line *= hist.max()
                    else:
                        kde_line *= (hist * np.diff(edges)).sum()

                if orientation in ('v', 'vertical'):
                    x = support
                    y = kde_line
                elif orientation in ('h', 'horizontal'):
                    x = kde_line
                    y = support

                if fill_kde is True:
                    if orientation in ('v', 'vertical'):
                        fill = 'tozeroy'
                    elif orientation in ('h', 'horizontal'):
                        fill = 'tozerox'
                elif isinstance(fill_kde, str):
                    fill = fill_kde
                else:
                    fill = None

                line = go.Scatter(x=x, y=y,
                                  legendgroup=key, name=key, fill=fill,
                                  mode='lines', line=line_kwargs)
                fig.add_traces(line, rows=row, cols=col)

        # Format plot on the way out
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        legend, tick_size, axis_label_size,
                                        axis_type='default',
                                        barmode=barmode, margin=margin,
                                        bargap=bargap, bargroupgap=bargroupgap,
                                        row=row, col=col)

        return fig

    def violin_plot(self,
                    arrays: Collection[np.ndarray],
                    neg_arrays: Collection[np.ndarray] = [],
                    keys: Collection[str] = [],
                    neg_keys: Collection[str] = [],
                    colors: Union[str, Collection[str]] = None,
                    neg_colors: Union[str, Collection[str]] = None,
                    alpha: float = 1.0,
                    orientation: str = 'v',
                    show_box: bool = False,
                    show_points: Union[str, bool] = False,
                    show_mean: bool = False,
                    spanmode: str = 'soft',
                    side: str = None,
                    add_cell_numbers: bool = True,
                    legend: bool = True,
                    figure: Union[go.Figure, go.FigureWidget] = None,
                    figsize: Tuple[int] = (None, None),
                    margin: str = 'auto',
                    title: str = None,
                    x_label: str = None,
                    y_label: str = None,
                    x_limit: Tuple[float] = None,
                    y_limit: Tuple[float] = None,
                    tick_size: float = None,
                    axis_label_size: float = None,
                    widget: bool = False,
                    row: int = None,
                    col: int = None,
                    **kwargs
                    ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly go.Figure object with violin distributions
        for each of the arrays given. If negative arrays are given,
        matches them with arrays and plots two distributions side
        by side.

        :param arrays: List of arrays to plot. Arrays are assumed to be
            one dimensional. If neg_arrays is given, arrays are plotted on
            the positive side.
        :param keys: Names corresponding to the data arrays. If not provided,
            the keys will be integers.
        :param neg_arrays: List of arrays to plot. Arrays are assumed to be
            1-dimensional. If neg_arrays is given, arrays are plotted on
            the positive side.
        :param neg_keys: Names corresponding to the neative data arrays. If not
            provided, the keys will be integers.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param neg_colors: Name of a color palette or map to use. Searches
            in seaborn/matplotlib first, then in Plotly to find the color map.
            If not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param alpha: Opacity of the fill color of the violin plots as a float
            in the range [0, 1].
        :param orientation: Orientation of the violin plot.
        :param show_box: If True, a box plot is made and overlaid over the
            violin plot.
        :param show_points: If True, individual data points are overlaid over
            the violin plot.
        :param show_mean: If True, dashed line is plotted at the mean value.
        :param spanmode: Determines how far the tails of the violin plot are
            extended. If 'hard', the plot spans as far as the data. If 'soft',
            the tails are extended.
        :param side: Side to plot the distribution. By default, the
            distribution is plotted on both sides, but can be 'positive'
            or 'negative' to plot on only one side.
        :param add_cell_numbers: If True, adds the number of cells to each key
            in keys.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Depending on name, passed to go.Violin or to
            go.Figure.update_traces(). The following kwargs are passed to
            go.Violin: 'bandwidth', 'fillcolor', 'hoverinfo', 'jitter', 'line',
            'marker', 'opacity', 'pointpos', 'points', 'span',
            'width', 'hoverinfo'

        :return: Figure object

        :raises AssertionError: If not all entries in arrays or neg_arrays are
            np.ndarray
        :raises AssertionError: If any entry in arrays or neg_arrays have more
            than one dimension.
        :raises AssertionError: If neg_arrays is given, and
            len(arrays) != len(neg_arrays)
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        # Format inputs
        violinmode = None
        arrays = self._format_arrays(arrays)
        neg_arrays = self._format_arrays(neg_arrays)
        arrays = [np.squeeze(a) for a in arrays]
        keys = self._format_keys(keys, default='dist', arrays=arrays,
                                 add_cell_numbers=add_cell_numbers)
        assert all([a.ndim in (1, 0) for a in arrays])
        if neg_arrays:
            assert len(arrays) == len(neg_arrays)
            neg_arrays = [np.squeeze(a) for a in neg_arrays]
            assert all([a.ndim in (1, 0) for a in neg_arrays])
            violinmode = 'overlay'
            side = 'positive'
        assert orientation in ('v', 'h', 'horizontal', 'vertical')
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays), alpha)
        neg_colors = self._build_colormap(neg_colors, len(neg_arrays))
        violin_kwargs = {k: v for k, v in kwargs.items()
                         if k in self._violin_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._violin_kwargs}
        meanline_kw = {'visible': show_mean, 'width': 3}

        # Build the figure and start plotting
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        for idx, (arr, key) in enumerate(zip(arrays, keys)):
            # Set the data key based on orientation
            if orientation in ('v', 'vertical'):
                y = arr
                x = None
                if neg_arrays:
                    neg_y = neg_arrays[idx]
                    neg_x = None
            elif orientation in ('h', 'horizontal'):
                y = None
                x = arr
                if neg_arrays:
                    neg_y = None
                    neg_x = neg_arrays[idx]

            # Set up the colors
            _c = next(colors)
            violin_kwargs.update({'fillcolor': _c})
            line = {'color': _c}

            # Make individual distributions on the plot
            trace = go.Violin(x=x, y=y, name=key, legendgroup=key, side=side,
                              spanmode=spanmode, box_visible=show_box,
                              points=show_points, hoverinfo='skip',
                              line=line, meanline=meanline_kw,
                              **violin_kwargs)
            fig.add_traces(trace, rows=row, cols=col)

            # Add the other half of the distributions if needed
            if neg_arrays:
                # Set up the colors
                _nc = next(neg_colors)
                violin_kwargs.update({'fillcolor': _nc})
                line = {'color': _nc}

                neg_side = 'negative' if side == 'positive' else 'positive'
                nkey = neg_keys[idx] if neg_keys else key
                neg_trace = go.Violin(x=neg_x, y=neg_y, side=neg_side,
                                      name=key, legendgroup=nkey,
                                      spanmode=spanmode, box_visible=show_box,
                                      points=show_points, hoverinfo='skip',
                                      line=line, meanline=meanline_kw,
                                      **violin_kwargs)
                fig.add_traces(neg_trace, rows=row, cols=col)

        # Format plot on the way out
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        legend, tick_size, axis_label_size,
                                        axis_type='default',
                                        violinmode=violinmode, margin=margin,
                                        row=row, col=col)
        fig.update_traces(**kwargs)

        return fig

    def ridgeline_plot(self,
                       arrays: Collection[np.ndarray],
                       keys: Collection[str] = [],
                       colors: Union[str, Collection[str]] = None,
                       spanmode: str = 'hard',
                       overlap: float = 3,
                       show_box: bool = False,
                       show_points: Union[str, bool] = False,
                       show_mean: bool = True,
                       add_cell_numbers: bool = True,
                       legend: bool = True,
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       margin: str = 'auto',
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       tick_size: float = None,
                       axis_label_size: float = None,
                       widget: bool = False,
                       row: int = None,
                       col: int = None,
                       **kwargs
                       ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly go.Figure object with partially overlapping
        distributions for each of the arrays given. Similar to a violin
        plot. See the section on ridgeline plots here for more information.

        :param arrays: List of arrays to create distributions from. Arrays
            are assumed to be one dimensional.
        :param keys: Names corresponding to the data arrays. If not provided,
            the keys will be integers.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param spanmode: Determines how far the tails of the violin plot are
            extended. If 'hard', the plot spans as far as the data. If 'soft',
            the tails are extended.
        :param overlap: Sets the amount of overlap between adjacent
            distributions. Larger values means more overlap.
        :param show_box: If True, a box plot is made and overlaid over the
            distribution.
        :param show_points: If True, individual data points are overlaid over
            the distribution.
        :param side: Side to plot the distribution. By default, the
            distribution is plotted on both sides, but can be 'positive'
            or 'negative' to plot on only one side.
        :param add_cell_numbers: If True, adds the number of cells to each key
            in keys.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Depending on name, passed to go.Violin or to
            go.Figure.update_traces(). The following kwargs are passed to
            go.Violin: 'bandwidth', 'fillcolor', 'hoverinfo', 'jitter', 'line',
            'marker', 'opacity', 'pointpos', 'points', 'span',
            'width', 'hoverinfo'

        :return: Figure object
        """
        # Plot the violin plots
        fig = self.violin_plot(arrays, keys=keys, colors=colors,
                               spanmode=spanmode, legend=legend,
                               show_box=show_box, show_points=show_points,
                               show_mean=show_mean, figure=figure, widget=widget,
                               add_cell_numbers=add_cell_numbers,
                               side='positive', orientation='h',
                               row=row, col=col, **kwargs)

        # Some settings for making a ridgeline out of the violin plot
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        legend, tick_size, axis_label_size,
                                        axis_type='default',
                                        xaxis_showgrid=False,
                                        xaxis_zeroline=False, margin=margin,
                                        row=row, col=col)
        fig.update_traces(width=overlap)

        return fig

    def heatmap_plot(self,
                     array: np.ndarray,
                     colorscale: str = 'viridis',
                     zmin: float = None,
                     zmid: float = None,
                     zmax: float = None,
                     robust_z: bool = False,
                     reverse: bool = False,
                     sort: str = None,
                     figure: Union[go.Figure, go.FigureWidget] = None,
                     figsize: Tuple[int] = (None, None),
                     margin: str = 'auto',
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     x_limit: Tuple[float] = None,
                     y_limit: Tuple[float] = None,
                     tick_size: float = None,
                     axis_label_size: float = None,
                     widget: bool = False,
                     row: int = None,
                     col: int = None,
                     **kwargs
                     ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly go.Figure object with a heatmap of the provided
        array. This function is just a very thin wrapper around go.Heatmap.

        :param array: Array from which to make the heatmap.
        :param colorscale: The colorscale to make the heatmap in. Options are
            more limited than the options for colors. Options include:
            "Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric",
            "Greens", "Greys", "Hot", "Jet", "Picnic", "Portland", "Rainbow",
            "RdBu", "Reds", "Viridis", "YlGnBu", and "YlOrRd".
        :param zmin: Sets the lower bound of the color domain. If given, zmax
            must also be given.
        :param zmid: Sets the midpoint of the color domain by setting zmin and
            zmax to be equidistant from this point.
        :param zmax: Sets the upper bound of the color domain. If given, zmin
            must also be given.
        :param robust_z: If True, uses percentiles to set zmin and zmax instead
            of extremes of the dataset.
        :param reverse: If True, the color mapping is reversed.
        :param sort: If the name of a distance metric, will sort the array
            according to that metric before producing the heatmap. Valid values
            are ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’,
            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’,
            ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’,
            ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
            ‘sokalsneath’, ‘sqeuclidean’, and ‘yule’.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Passed to go.Heatmap.

        :return: Figure object.

        :raises AssertionError: If array dtype is object.
        :raises AssertionError: If array is not two-dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        assert array.dtype != 'object'
        assert array.ndim == 2
        assert len(figsize) == 2

        # Build the figure and make the heatmap
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        # Similar to how seaborn determines robust quantiles
        if robust_z:
            zmin = np.nanpercentile(array, 2)
            zmax = np.nanpercentile(array, 98)
            zmid = None

        if sort:
            darr = metrics.pairwise_distances(array, metric=sort)
            idx = darr[0, :].argsort()
            array = array[idx]

        trace = go.Heatmap(z=array, zmin=zmin, zmax=zmax,
                           zmid=zmid, colorscale=colorscale,
                           reversescale=reverse, **kwargs)
        fig.add_traces(trace, rows=row, cols=col)

        # None is for the legend kwarg
        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        None, tick_size, axis_label_size,
                                        axis_type='noline', margin=margin,
                                        row=row, col=col)

        return fig

    def heatmap2d_plot(self,
                       x_array: np.ndarray,
                       y_array: np.ndarray,
                       colorscale: str = 'viridis',
                       zmin: float = None,
                       zmid: float = None,
                       zmax: float = None,
                       robust_z: bool = False,
                       xbinsize: float = None,
                       ybinsize: float = None,
                       histfunc: str = 'count',
                       histnorm: str = "",
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       margin: str = 'auto',
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       tick_size: float = None,
                       axis_label_size: float = None,
                       widget: bool = False,
                       row: int = None,
                       col: int = None,
                       **kwargs
                       ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly go.Figure object with a two dimensional density
        heatmap of the provided arrays. This function is just a very
        thin wrapper around go.Heatmap2d.

        :param x_array: Array containing observations for the data on the
            x-axis. Expected to be one dimensional.
        :param y_array: Array containing observations for the data on the
            y-axis. Expected to be one dimensional.
        :param colorscale: The colorscale to make the heatmap in. Options are
            more limited than the options for colors. Options include:
            "Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric",
            "Greens", "Greys", "Hot", "Jet", "Picnic", "Portland", "Rainbow",
            "RdBu", "Reds", "Viridis", "YlGnBu", and "YlOrRd".
        :param zmin: Sets the lower bound of the color domain. If given, zmax
            must also be given.
        :param zmid: Sets the midpoint of the color domain by setting zmin and
            zmax to be equidistant from this point.
        :param zmax: Sets the upper bound of the color domain. If given, zmin
            must also be given.
        :param robust_z: If True, uses percentiles to set zmin and zmax instead
            of extremes of the dataset.
        :param xbinsize: Size of the bins along the x-axis.
        :param ybinsize: Size of the bins along the y-axis.
        :param histfunc: Specifies the binning function used for
            this histogram trace. If “count”, the histogram values
            are computed by counting the number of values lying
            inside each bin. Can also be “sum”, “avg”, “min”, “max”.
        :param histnorm: Specifies the type of normalization used
            for this histogram trace. If “”, the span of each bar
            corresponds to the number of occurrences  If “percent” /
            “probability”, the span of each bar corresponds to the
            percentage / fraction of occurrences with respect to
            the total number of sample points (here, the sum of
            all bin HEIGHTS equals 100% / 1). If “density”, the
            span of each bar corresponds to the number of
            occurrences in a bin divided by the size of the bin
            interval. If probability density, the area of each
            bar corresponds to the probability that an event will
            fall into the corresponding bin (here, the sum of all
            bin AREAS equals 1).
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot.
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Passed to go.Histogram2d.

        :return: Figure object

        :raises AssertionError: If x_array or y_array are more than one
            dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        assert np.squeeze(x_array).ndim in (1, 0)
        assert np.squeeze(y_array).ndim in (1, 0)
        assert len(figsize) == 2

        # Similar to how seaborn determines robust quantiles
        if robust_z:
            _arr = np.stack((x_array.ravel(), y_array.ravel()))
            zmin = np.nanpercentile(_arr, 2)
            zmax = np.nanpercentile(_arr, 98)
            zmid = None

        # Build the figure and plot the density histogram
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        trace = go.Histogram2d(x=x_array, y=y_array,
                               colorscale=colorscale,
                               histfunc=histfunc,
                               histnorm=histnorm,
                               zmin=zmin, zmid=zmid, zmax=zmax,
                               xbins=dict(size=xbinsize),
                               ybins=dict(size=ybinsize),
                               **kwargs)
        fig.add_traces(trace, rows=row, cols=col)

        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        tick_size, axis_label_size,
                                        axis_type='noline', margin=margin,
                                        row=row, col=col)

        return fig

    def contour2d_plot(self,
                       x_array: np.ndarray,
                       y_array: np.ndarray,
                       colorscale: str = 'viridis',
                       fill: bool = True,
                       zmin: float = None,
                       zmid: float = None,
                       zmax: float = None,
                       robust_z: bool = False,
                       width: float = 0.5,
                       xbinsize: float = None,
                       ybinsize: float = None,
                       histfunc: str = 'count',
                       histnorm: str = "",
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       margin: str = 'auto',
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       tick_size: float = None,
                       axis_label_size: float = None,
                       widget: bool = False,
                       row: int = None,
                       col: int = None,
                       **kwargs
                       ) -> Union[go.Figure, go.FigureWidget]:
        """
        Builds a Plotly go.Figure object with a two dimensional density
        contour heatmap of the provided arrays. This function is just a very
        thin wrapper around go.Heatmap2dContour.

        :param x_array: Array containing observations for the data on the
            x-axis. Expected to be one dimensional.
        :param y_array: Array containing observations for the data on the
            y-axis. Expected to be one dimensional.
        :param colorscale: The colorscale to make the heatmap in. Options are
            more limited than the options for colors. Options include:
            "Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric",
            "Greens", "Greys", "Hot", "Jet", "Picnic", "Portland", "Rainbow",
            "RdBu", "Reds", "Viridis", "YlGnBu", and "YlOrRd".
        :param fill: If True, space between contour lines is filled with color,
            otherwise, only the lines are colored.
        :param zmin: Sets the lower bound of the color domain. If given, zmax
            must also be given.
        :param zmid: Sets the midpoint of the color domain by setting zmin and
            zmax to be equidistant from this point.
        :param zmax: Sets the upper bound of the color domain. If given, zmin
            must also be given.
        :param robust_z: If True, uses percentiles to set zmin and zmax instead
            of extremes of the dataset.
        :param width: Width of the contour lines.
        :param xbinsize: Size of the bins along the x-axis.
        :param ybinsize: Size of the bins along the y-axis.
        :param histfunc: Specifies the binning function used for
            this histogram trace. If “count”, the histogram values
            are computed by counting the number of values lying
            inside each bin. Can also be “sum”, “avg”, “min”, “max”.
        :param histnorm: Specifies the type of normalization used
            for this histogram trace. If “”, the span of each bar
            corresponds to the number of occurrences  If “percent” /
            “probability”, the span of each bar corresponds to the
            percentage / fraction of occurrences with respect to
            the total number of sample points (here, the sum of
            all bin HEIGHTS equals 100% / 1). If “density”, the
            span of each bar corresponds to the number of
            occurrences in a bin divided by the size of the bin
            interval. If probability density, the area of each
            bar corresponds to the probability that an event will
            fall into the corresponding bin (here, the sum of all
            bin AREAS equals 1).
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple
            of length two. To leave height or width unchanged, set as None.
        :param margin: Set the size of the margins. If 'auto', all margins
            are set to defualt values. If 'zero' or 'tight', margins are
            removed. If a number is given, sets all margins to that number.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param tick_size: Size of the font of the axis tick labels.
        :param axis_label_size: Size of the font of the axis label.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param row: If Figure has multiple subplots, specifies which row
            to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param col: If Figure has multiple subplots, specifies which
            col to use for the plot. Indexing starts at 1. Note that some
            formatting args (such as figsize) may still be applied to all
            subplots. Both row and col must be provided together.
        :param kwargs: Passed to go.Heatmap2dContour

        :return: Figure object

        :raises AssertionError: If x_array or y_array are more than one
            dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        assert np.squeeze(x_array).ndim in (1, 0)
        assert np.squeeze(y_array).ndim in (1, 0)
        assert len(figsize) == 2

        # Similar to how seaborn determines robust quantiles
        if robust_z:
            _arr = np.stack((x_array.ravel(), y_array.ravel()))
            zmin = np.nanpercentile(_arr, 2)
            zmax = np.nanpercentile(_arr, 98)
            zmid = None

        if fill:
            contours = {'coloring': 'fill'}
        else:
            contours = {'coloring': 'lines'}
        line = {'width': width}

        # Build the figure and plot the contours
        if figure:
            fig = figure
        elif widget:
            fig = go.FigureWidget()
        else:
            fig = go.Figure()

        trace = go.Histogram2dContour(x=x_array, y=y_array,
                                      colorscale=colorscale,
                                      contours=contours, line=line,
                                      histfunc=histfunc,
                                      histnorm=histnorm,
                                      zmin=zmin, zmid=zmid, zmax=zmax,
                                      xbins=dict(size=xbinsize),
                                      ybins=dict(size=ybinsize),
                                      **kwargs)
        fig.add_traces(trace, rows=row, cols=col)

        fig = self._apply_format_figure(fig, figsize, title,
                                        x_label, y_label, x_limit, y_limit,
                                        tick_size, axis_label_size,
                                        axis_type='noline', margin=margin,
                                        row=row, col=col)

        return fig

    def trace_color_plot(self,
                         trace_array: np.ndarray,
                         color_arrays: Collection[np.ndarray] = [],
                         color_thres: Collection[float] = [],
                         colors: Union[str, Collection[str]] = None,
                         rows: int = 6,
                         cols: int = 4,
                         max_figures: int = None,
                         time: np.ndarray = None,
                         figsize: Tuple[float] = (None, None),
                         title: str = None,
                         x_label: str = None,
                         y_label: str = None,
                         x_limit: Tuple[float] = None,
                         y_limit: Tuple[float] = None,
                         **kwargs
                         ) -> Generator:
        """
        Generates Plotly go.Figure objects with subplots for each individual
        trace in trace_array. Traces can be colored with discrete colors based
        on arbitrary criteria in color_arrays. For example, this function can
        be used to evaluate the success of peak segmentation by passing the
        traces to trace_array, and the peak segmentation to color_arrays.

        :param trace_array: Array containing the values to be plotted. Assumed
            structure is two-dimensional of shape n_cells x n_features.
        :param color_arrays: Collection of arrays of the same shape as
            trace_array. Used with color_thres to determine what sections
            of the trace should be colored. There is no limit on the number
            of arrays passed, but color_thres must be the same length.
        :param color_thres: Collection of thresholds associated with
            color_arrays. For each array and threshold, the trace will be
            colored wherever color_array >= color_thres.
        :param colors: Name of a color palette or map to use. Will use the
            first color as the base color of trace, and subsequent colors
            for each color_array. Searches first in seaborn/matplotlib,
            then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param rows: Number of rows of subplots to make for each figure.
        :param cols: Number of columns of subplots to make for each figure.
        :param max_figures: Maximum number of figures to produce. If None,
            makes enough figures to plot all of the traces.
        :param time: Time axis for the plot. Must be the same size as the
            second dimension of arrays.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object.
        :param kwargs: Depending on name, passed to the "line" keyword
            argument of go.Scatter or as keyword arguments for go.Scatter.
            The following kwargs are passed to "line": 'color', 'dash',
            'shape', 'simplify', 'smoothing', 'width', 'hoverinfo'

        :return: Generator that produces go.Figure objects

        :raises AssertionError: If any array in color_arrays does not
            have the same shape as trace array.
        :raises AssertionError: If length of color_arrays does not
            equal length of color_thres.
        """
        # Check inputs
        color_arrays = self._format_arrays(color_arrays)
        color_thres = self._format_arrays(color_thres)
        assert all([trace_array.shape == c.shape for c in color_arrays])
        assert len(color_arrays) == len(color_thres)

        colors = self._build_colormap(colors, len(color_arrays) + 1)
        time = time if time else np.arange(trace_array.shape[1])
        line_kwargs = {k: v for k, v in kwargs.items()
                       if k in self._line_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._line_kwargs}

        # Set up the figure layout
        num_traces = trace_array.shape[0]
        num_subplts = int(np.ceil(num_traces / (rows * cols)))
        if max_figures and max_figures < num_subplts:
            num_subplts = max_figures

        # Iterate through all of the traces
        trace_idx = 0
        for _ in range(num_subplts):
            fig = psubplt.make_subplots(rows=rows, cols=cols)
            for fidx in range(rows * cols):
                try:
                    trace = trace_array[trace_idx]
                    r = fidx // cols + 1
                    c = fidx % cols + 1

                    # Plot the trace first and plot the others on top
                    line_kwargs.update({'color': next(colors)})
                    background = go.Scatter(x=time, y=trace, line=line_kwargs,
                                            showlegend=False, mode='lines',
                                            **kwargs)
                    fig.add_traces(background, row=r, col=c)

                    # Plot regions of the traces that will be different colors
                    for carr, thres in zip(color_arrays, color_thres):
                        carr = carr[trace_idx]
                        color_trace = np.where(carr >= thres, trace, np.nan)

                        line_kwargs.update({'color': next(colors)})
                        ctrace = go.Scatter(x=time, y=color_trace,
                                            line=line_kwargs,
                                            showlegend=False, mode='lines',
                                            **kwargs)
                        fig.add_traces(ctrace, row=r, col=c)

                    trace_idx += 1
                except IndexError:
                    # Reached the end of the traces
                    break

            fig = self._apply_format_figure(fig, figsize, title,
                                            x_label, y_label, x_limit, y_limit,
                                            axis_type='default')

            yield fig
