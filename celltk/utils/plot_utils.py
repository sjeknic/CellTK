import itertools
import functools
from typing import Collection, Union, Callable, Generator, Tuple

import numpy as np
import sklearn.base as base
import sklearn.preprocessing as preproc
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
        'title': dict(font=dict(color='#242424', size=24)),
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

    @staticmethod
    def _format_colors(color: str, alpha: float = None) -> str:
        """Converst hexcode colors to RGBA to allow transparency"""
        if isinstance(color, (list, tuple)):
            if all([isinstance(f, (float, int)) for f in color]):
                # Assume rgb
                if alpha: color += (alpha, )
                else: color += (1.,)
                color_str = str(tuple([c for c in color]))
            else:
                # Assume first value is alpha
                alpha = alpha if alpha else color[0]
                if alpha < 0.125: alpha = 0.125
                values = pcol.unlabel_rgb(color[1]) + (alpha,)
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
                color_str = str(tuple([c for c in color]))
                if len(color) == 4:
                    return f'rgba{color_str}'
                else:
                    return f'rgb{color_str}'
            elif color[:3] in ('rgb'):
                if alpha:
                    # extract the rgb values
                    vals = pcol.unlabel_rgb(color)
                    vals += (alpha, )
                    color_str = str(tuple([v for v in vals]))
                    color = f'rgba{color_str}'
                    return color
            else:
                try:
                    vals = mcolors.to_rgba(color)
                    if alpha:
                        vals = (*vals[:-1], alpha)
                    color_str = str(tuple([v for v in vals]))
                    color = f'rgba{color_str}'
                    return color
                except ValueError:
                    raise ValueError(f'Did not understand color {color}')

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
                  time: Union[Collection[np.ndarray], np.ndarray] = None,
                  legend: bool = True,
                  figure: Union[go.Figure, go.FigureWidget] = None,
                  figsize: Tuple[int] = (None, None),
                  title: str = None,
                  x_label: str = None,
                  y_label: str = None,
                  x_limit: Tuple[float] = None,
                  y_limit: Tuple[float] = None,
                  widget: bool = False,
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
        :param time: Time axis for the plot. Must be the same size as the
            second dimension of arrays.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param kwargs: Depending on name, passed to the "line" keyword
            argument of go.Scatter or as keyword arguments for go.Scatter.
            The following kwargs are passed to "line": 'color', 'dash',
            'shape', 'simplify', 'smoothing', 'width', 'hoverinfo'

        :return: Figure object

        :raises AssertionError: If not all items in arrays are np.ndarray.
        :raises AssertionError: If any item in arrays is not two dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        :raises TypeError: If time is not an np.ndarray or collection of
            np.ndarray.
        """
        # Format inputs
        assert all([isinstance(a, np.ndarray) for a in arrays])
        assert all([a.ndim == 2 for a in arrays])
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays))
        if normalizer: normalizer = self._build_normalizer_func(normalizer)
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        line_kwargs = {k: v for k, v in kwargs.items()
                       if k in self._line_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._line_kwargs}

        # Build the figure and start plotting
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()

        for idx, (arr, key) in enumerate(itertools.zip_longest(arrays, keys)):
            # Get the key
            if not key:
                key = f'line_{idx}'
            key += f' | n={arr.shape[0]}'

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
                                f'type {type(time)}')

            lines = []
            _legend = True
            for a, y in enumerate(arr):
                if a: _legend = False
                line_kwargs.update({'color': next(colors)})

                lines.append(
                    go.Scatter(x=x, y=y, legendgroup=key, name=key,
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
                        go.Scatter(x=np.hstack([x, x[::-1]]),
                                   y=np.hstack([hi, lo[::-1]]), fill='tozerox',
                                   fillcolor=self._format_colors(line_kwargs['color'], 0.25),
                                   showlegend=False, legendgroup=key,
                                   name=key, line=dict(color='rgba(255,255,255,0)'),
                                   hoverinfo='skip')
                    )

            fig.add_traces(lines)

        # Upate the axes and figure layout
        self._default_axis_layout['title'].update({'text': x_label})
        fig.update_xaxes(**self._default_axis_layout)
        self._default_axis_layout['title'].update({'text': y_label})
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_layout(template=self._template,
                          title=title,
                          xaxis_range=x_limit,
                          yaxis_range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

        return fig

    def scatter_plot(self,
                     x_arrays: Collection[np.ndarray] = [],
                     y_arrays: Collection[np.ndarray] = [],
                     keys: Collection[str] = [],
                     estimator: Union[Callable, str, functools.partial] = None,
                     err_estimator: Union[Callable, str, functools.partial] = None,
                     normalizer: Union[Callable, str] = None,
                     colors: Union[str, Collection[str]] = None,
                     alpha: float = 1.0,
                     symbols: Union[str, Collection[str]] = None,
                     legend: bool = True,
                     figure: Union[go.Figure, go.FigureWidget] = None,
                     figsize: Tuple[int] = (None, None),
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     x_limit: Tuple[float] = None,
                     y_limit: Tuple[float] = None,
                     widget: bool = False,
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
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param alpha: Opacity of the marker fill colors.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
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
        """
        # Format inputs - should be cells x features
        if x_arrays and y_arrays: assert len(x_arrays) == len(y_arrays)
        assert all(isinstance(a, np.ndarray) for a in x_arrays)
        assert all(isinstance(a, np.ndarray) for a in y_arrays)
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
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()
        traces = []
        zipped = itertools.zip_longest(x_arrays, y_arrays, keys,
                                       fillvalue=None)
        for idx, (xarr, yarr, key) in enumerate(zipped):
            # Get the key
            if not key:
                key = f'group_{idx}'
            n = yarr.shape[0] if yarr is not None else xarr.shape[0]
            key += f' | n={n}'

            # err_estimator is used to set error bars
            if err_estimator:
                err_arr = err_estimator(yarr)
            else:
                err_arr = None

            # estimator is used to condense all the cells to a single point
            if estimator:
                yarr = estimator(yarr)

            # normalizer is used to get data onto the same scale
            if normalizer:
                yarr = normalizer(yarr.reshape(-1, 1)).reshape(yarr.shape)
                if err_arr is not None:
                    err_arr = normalizer(
                        err_arr.reshape(-1, 1), scale_only=True
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
                    err_plus = y - err_arr[0, :]
                    err_minus = err_arr[-1, :] - y
                    error_y.update({'array': err_plus,
                                    'arrayminus': err_minus})

            marker_kwargs.update(dict(color=next(colors),
                                      symbol=next(symbols)))
            traces.append(
                go.Scatter(x=x, y=y, legendgroup=key, name=key,
                           showlegend=legend, mode='markers',
                           error_x=error_x, error_y=error_y,
                           marker=marker_kwargs, **kwargs)
            )

        fig.add_traces(traces)

        self._default_axis_layout['title'].update({'text': x_label})
        fig.update_xaxes(**self._default_axis_layout)
        self._default_axis_layout['title'].update({'text': y_label})
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_layout(template=self._template,
                          xaxis_range=x_limit,
                          yaxis_range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

        return fig

    def bar_plot(self,
                 arrays: Collection[np.ndarray],
                 keys: Collection[str] = [],
                 estimator: Union[Callable, str, functools.partial] = None,
                 err_estimator: Union[Callable, str, functools.partial] = None,
                 ax_labels: Collection[str] = None,
                 colors: Union[str, Collection[str]] = None,
                 orientation: str = 'v',
                 barmode: str = 'group',
                 legend: bool = True,
                 figure: Union[go.Figure, go.FigureWidget] = None,
                 figsize: Tuple[int] = (None, None),
                 title: str = None,
                 x_label: str = None,
                 y_label: str = None,
                 x_limit: Tuple[float] = None,
                 y_limit: Tuple[float] = None,
                 widget: bool = False,
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
        :param ax_labels: Labels for the categorical axis.
        :param orientation: Orientation of the bar plot.
        :param barmode: Keyword argument describing how to group the bars.
            Options are 'group', 'overlay', 'relative', and 'stack'.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is vertical.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param kwargs: Depending on name, passed to go.Bar or to
            go.Figure.update_traces(). The following kwargs are passed to
            go.Bar: 'hoverinfo', 'marker', 'width'.

        :return: Figure object

        :raises AssertionError: If not all items in arrays are np.ndarray.
        :raises AssertionError: If orientation is a disallowed value.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        # Format data
        assert all([isinstance(a, np.ndarray) for a in arrays])
        assert orientation in ('v', 'h', 'horizontal', 'vertical')
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays))
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        bar_kwargs = {k: v for k, v in kwargs.items()
                      if k in self._bar_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._bar_kwargs}

        # Build the figure and start plotting
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()
        for idx, (arr, key) in enumerate(itertools.zip_longest(arrays, keys)):
            # Get the key
            if not key:
                key = f'bar_{idx}'
            key += f' | n={arr.shape[0]}'

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
                    if err_arr.ndim == 1:
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
                    if err_arr.ndim == 1:
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
            fig.add_trace(trace)

        # Format plot on the way out
        fig.update_traces(**kwargs)
        fig.update_layout(template=self._template, barmode=barmode,
                          title=title)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

        return fig

    def histogram_plot(self,
                     arrays: Collection[np.ndarray],
                     keys: Collection[str] = [],
                     histfunc: str = 'count',
                     histnorm: str = "",
                     nbins: int = None,
                     binsize: float = None,
                     bargap: float = None,
                     bargroupgap: float = None,
                     cumulative: bool = False,
                     colors: Union[str, Collection[str]] = None,
                     alpha: float = 1.0,
                     orientation: str = 'v',
                     barmode: str = 'group',
                     legend: bool = True,
                     figure: Union[go.Figure, go.FigureWidget] = None,
                     figsize: Tuple[int] = (None, None),
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     x_limit: Tuple[float] = None,
                     y_limit: Tuple[float] = None,
                     widget: bool = False,
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
            interval. If probability density, the area of each
            bar corresponds to the probability that an event will
            fall into the corresponding bin (here, the sum of all
            bin AREAS equals 1).
        :param nbins: Maximum number of bins allowed. Ignored if
            binsize is set.
        :param binsize: Size of each bin.
        :param bargap: Gap between bars in adjacent locations.
        :param bargroupgap: Gap between bars in the same location.
        :param cumulative: If True, the histogram will plot cumulative
            occurances.
        :param colors: Name of a color palette or map to use. Searches first
            in seaborn/matplotlib, then in Plotly to find the color map. If
            not provided, the color map will be glasbey. Can also be list
            of named CSS colors or hexadecimal or RGBA strings.
        :param alpha: Opacity of the fill color of the histogram as a float
            in the range [0, 1].
        :param orientation: Orientation of the bar plot.
        :param barmode: Keyword argument describing how to group the bars.
            Options are 'group', 'overlay', 'stack', and 'relative'.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is vertical.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param kwargs: Depending on name, passed to go.Bar or to
            go.Figure.update_traces(). The following kwargs are passed to
            go.Bar: 'hoverinfo', 'marker', 'width'.

        :return: Figure object

        :raises AssertionError: If not all items in arrays are np.ndarray.
        :raises AssertionError: If orientation is a disallowed value.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        # Format data
        assert all([isinstance(a, np.ndarray) for a in arrays])
        assert orientation in ('v', 'h', 'horizontal', 'vertical')
        assert len(figsize) == 2

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays), alpha)

        # Build the figure and start plotting
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()

        for idx, (arr, key) in enumerate(itertools.zip_longest(arrays, keys)):
            # Get the key
            if not key:
                key = f'bar_{idx}'
            key += f' | n={arr.shape[0]}'

            if orientation in ('v', 'vertical'):
                y = None
                x = arr
            elif orientation in ('h', 'horizontal'):
                y = arr
                x = None

            # Set up the colors
            _c = next(colors)
            marker_kwargs = {'color': _c, 'line': {'color': _c}}
            cum_kwargs = {'enabled': cumulative}

            # Make individual distributions on the plot
            trace = go.Histogram(x=x, y=y, name=key, legendgroup=key,
                                 histfunc=histfunc, histnorm=histnorm,
                                 orientation=orientation,
                                 cumulative=cum_kwargs,
                                 nbinsx=nbins, nbinsy=nbins,
                                 xbins=dict(size=binsize),
                                 ybins=dict(size=binsize),
                                 marker=marker_kwargs,
                                 **kwargs)
            fig.add_trace(trace)

        # Format plot on the way out
        fig.update_traces(**kwargs)
        fig.update_layout(template=self._template, barmode=barmode,
                          title=title, bargap=bargap, bargroupgap=bargroupgap)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

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
                    legend: bool = True,
                    figure: Union[go.Figure, go.FigureWidget] = None,
                    figsize: Tuple[int] = (None, None),
                    title: str = None,
                    x_label: str = None,
                    y_label: str = None,
                    x_limit: Tuple[float] = None,
                    y_limit: Tuple[float] = None,
                    widget: bool = False,
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
        :param spanmode: Determines how far the tails of the violin plot are
            extended. If 'hard', the plot spans as far as the data. If 'soft',
            the tails are extended.
        :param side: Side to plot the distribution. By default, the
            distribution is plotted on both sides, but can be 'positive'
            or 'negative' to plot on only one side.
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
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
        assert all([isinstance(a, np.ndarray) for a in arrays])
        arrays = [np.squeeze(a) for a in arrays]
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
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()
        for idx, (arr, key) in enumerate(itertools.zip_longest(arrays, keys)):
            # Get the key
            if not key:
                key = f'dist_{idx}'
            key += f' | n={arr.shape[0]}'

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
            fig.add_trace(trace)

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
                fig.add_trace(neg_trace)

        # Format plot on the way out
        fig.update_traces(**kwargs)
        fig.update_layout(template=self._template,
                          violinmode=violinmode,
                          title=title)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

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
                       legend: bool = True,
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       widget: bool = False,
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
        :param legend: If False, no legend is made on the plot.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
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
                               side='positive', orientation='h', **kwargs)

        # Some settings for making a ridgeline out of the violin plot
        fig.update_traces(width=overlap)
        fig.update_layout(template=self._template,
                          title=title, xaxis_showgrid=False,
                          xaxis_zeroline=False)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

        return fig

    def heatmap_plot(self,
                     array: np.ndarray,
                     colorscale: str = 'viridis',
                     zmin: float = None,
                     zmid: float = None,
                     zmax: float = None,
                     reverse: bool = False,
                     figure: Union[go.Figure, go.FigureWidget] = None,
                     figsize: Tuple[int] = (None, None),
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     x_limit: Tuple[float] = None,
                     y_limit: Tuple[float] = None,
                     widget: bool = False,
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
        :param reverse: If True, the color mapping is reversed.
        :param figure: If a go.Figure object is given, will be used to make
            the plot instead of a blank figure.
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param kwargs: Passed to go.Heatmap.

        :return: Figure object.

        :raises AssertionError: If array is not two-dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        assert array.ndim == 2
        assert len(figsize) == 2

        # Build the figure and make the heatmap
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()
        trace = go.Heatmap(z=array, zmin=zmin, zmax=zmax,
                           zmid=zmid, colorscale=colorscale,
                           reversescale=reverse, **kwargs)
        fig.add_trace(trace)

        fig.update_layout(template=self._template, title=title)
        fig.update_xaxes(**self._no_line_axis)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._no_line_axis)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

        return fig

    def heatmap2d_plot(self,
                       x_array: np.ndarray,
                       y_array: np.ndarray,
                       colorscale: str = 'viridis',
                       zmin: float = None,
                       zmid: float = None,
                       zmax: float = None,
                       xbinsize: float = None,
                       ybinsize: float = None,
                       histfunc: str = 'count',
                       histnorm: str = "",
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       widget: bool = False,
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
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot.
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param kwargs: Passed to go.Histogram2d.

        :return: Figure object

        :raises AssertionError: If x_array or y_array are more than one
            dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        assert np.squeeze(x_array).ndim in (1, 0)
        assert np.squeeze(y_array).ndim in (1, 0)
        assert len(figsize) == 2

        # Build the figure and plot the density histogram
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()
        trace = go.Histogram2d(x=x_array, y=y_array,
                               colorscale=colorscale,
                               histfunc=histfunc,
                               histnorm=histnorm,
                               zmin=zmin, zmid=zmid, zmax=zmax,
                               xbins=dict(size=xbinsize),
                               ybins=dict(size=ybinsize),
                               **kwargs)
        fig.add_trace(trace)

        fig.update_layout(template=self._template, title=title)
        fig.update_xaxes(**self._no_line_axis)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._no_line_axis)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

        return fig

    def contour2d_plot(self,
                       x_array: np.ndarray,
                       y_array: np.ndarray,
                       colorscale: str = 'viridis',
                       zmin: float = None,
                       zmid: float = None,
                       zmax: float = None,
                       xbinsize: float = None,
                       ybinsize: float = None,
                       histfunc: str = 'count',
                       histnorm: str = "",
                       figure: Union[go.Figure, go.FigureWidget] = None,
                       figsize: Tuple[int] = (None, None),
                       title: str = None,
                       x_label: str = None,
                       y_label: str = None,
                       x_limit: Tuple[float] = None,
                       y_limit: Tuple[float] = None,
                       widget: bool = False,
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
        :param zmin: Sets the lower bound of the color domain. If given, zmax
            must also be given.
        :param zmid: Sets the midpoint of the color domain by setting zmin and
            zmax to be equidistant from this point.
        :param zmax: Sets the upper bound of the color domain. If given, zmin
            must also be given.
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
        :param figsize: Height and width of the plot in pixels. Must be a tuple of
            length two. To leave height or width unchanged, set as None.
        :param title: Title to add to the plot
        :param x_label: Label of the x-axis
        :param y_label: Label of the y-axis
        :param x_limit: Initial limits for the x-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is horizontal.
        :param y_limit: Initial limits for the y-axis. Can be changed if
            the plot is saved as an HTML object. Only applies if orientation
            is veritcal.
        :param widget: If True, returns a go.FigureWidget object instead of
            a go.Figure object.
        :param kwargs: Passed to go.Heatmap2dContour

        :return: Figure object

        :raises AssertionError: If x_array or y_array are more than one
            dimensional.
        :raises AssertionError: If figsize is not a tuple of length two.
        """
        assert np.squeeze(x_array).ndim in (1, 0)
        assert np.squeeze(y_array).ndim in (1, 0)
        assert len(figsize) == 2

        # Build the figure and plot the contours
        if figure: fig = figure
        elif widget: fig = go.FigureWidget()
        else: fig = go.Figure()
        trace = go.Histogram2dContour(x=x_array, y=y_array,
                                      colorscale=colorscale,
                                      histfunc=histfunc,
                                      histnorm=histnorm,
                                      zmin=zmin, zmid=zmid, zmax=zmax,
                                      xbins=dict(size=xbinsize),
                                      ybins=dict(size=ybinsize),
                                      **kwargs)
        fig.add_trace(trace)

        fig.update_layout(template=self._template, title=title)
        fig.update_xaxes(**self._no_line_axis)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._no_line_axis)
        fig.update_yaxes(title=y_label, range=y_limit)

        # Set size only if not None, so as to not overwrite previous changes
        h, w = figsize
        if h: fig.update_layout(height=h)
        if w: fig.update_layout(width=w)

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
                    fig.add_trace(background, row=r, col=c)

                    # Plot regions of the traces that will be different colors
                    for carr, thres in zip(color_arrays, color_thres):
                        carr = carr[trace_idx]
                        color_trace = np.where(carr >= thres, trace, np.nan)

                        line_kwargs.update({'color': next(colors)})
                        ctrace = go.Scatter(x=time, y=color_trace,
                                            line=line_kwargs,
                                            showlegend=False, mode='lines',
                                            **kwargs)
                        fig.add_trace(ctrace, row=r, col=c)

                    trace_idx += 1
                except IndexError:
                    # Reached the end of the traces
                    break

            yield fig
