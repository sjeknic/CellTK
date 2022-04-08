import itertools
import functools
import inspect
from typing import Collection, Union, Callable, Generator, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as pltcol
import plotly.graph_objects as go
import plotly.colors as pcol
import colorcet as cc

import celltk.utils.estimator_utils


class PlotHelper:

    '''
    Okay, I don't know what the fuck I'm doing anymore. I have to do something, but
    don't want to do anything and have no fucking clue how to structure this. I know
    as soon as I start, I'll fuck it up and realize I should have done it differently.

    Anyways, the general steps are:
    1) User provides the data arrays to plot
    2) User provides the type of plot
    3) User provides colors, labels, and other figure params

    3.9) Plotter saves information about input data (e.g. cell nums)
    4) Plotter formats data arrays as needed
        - Applies estimator
        - Applies error estimator
        - Collects cells, etc.
    5) Plotter starts figure and applies lines
    6) Plotter returns figures
    '''
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
        'tickfont': dict(color=_ax_col, size=12, family=_font_families),
        'ticklen': 7,
        'ticks': 'outside',
        'tickwidth': 2,
        'title': dict(font=dict(color='#242424', size=20)),
        'zeroline': False,
    }
    _no_line_axis = _default_axis_layout.copy()
    _no_line_axis['showline'] = False
    _default_legend_layout = {
        'borderwidth': 0,
        'font': dict(color=_ax_col, size=12, family=_font_families),
        'itemsizing': 'constant',
        'itemwidth': 30,  # must be >= 30
        'title': {'font': dict(color=_ax_col, size=12,
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
    _violin_kwargs = ('bandwidth', 'fillcolor', 'hoverinfo', 'jitter', 'line',
                      'marker', 'opacity', 'pointpos', 'points', 'span',
                      'width', 'hoverinfo')
    _bar_kwargs = ('hoverinfo', 'marker', 'width')

    def _build_colormap(self,
                        colors: Union[str, Collection[str]],
                        number: int
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
                    # Try getting it from plotly
                    _col = pcol.get_colorscale(colors)
                    _col = pcol.colorscale_to_colors(_col)
                except Exception:  # This will be PlotlyError, find it
                    raise ValueError(f'Could not get colorscale for {colors}')

        _col = [self._format_colors(c) for c in _col]
        return itertools.cycle(_col)

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
                color = pcol.hex_to_rgb(color)
                if alpha:
                    color += (alpha,)
                color_str = str(tuple([c for c in color]))
                if len(color) == 4:
                    return f'rgba{color_str}'
                else:
                    return f'rgb{color_str}'

            else:
                assert color[:3] in ('rgb')  # only rgb for now I think
                if alpha:
                    # extract the rgb values
                    vals = pcol.unlabel_rgb(color)
                    vals += (alpha, )
                    color_str = str(tuple([v for v in vals]))
                    color = f'rgba{color_str}'

                return color

    def _build_estimator_func(self,
                              func: Union[Callable, str, functools.partial],
                              *args, **kwargs
                              ) -> functools.partial:
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

    def line_plot(self,
                  arrays: Collection[np.ndarray],
                  keys: Collection[str] = [],
                  estimator: Union[Callable, str, functools.partial] = None,
                  err_estimator: Union[Callable, str, functools.partial] = None,
                  colors: Union[str, Collection[str]] = None,
                  time: np.ndarray = None,
                  legend: bool = True,
                  figure: go.Figure = None,
                  title: str = None,
                  x_label: str = None,
                  y_label: str = None,
                  x_limit: Tuple[float] = None,
                  y_limit: Tuple[float] = None,
                  *args, **kwargs
                  ) -> Union[go.Figure, go.FigureWidget]:
        """"""
        # Format inputs
        assert all([isinstance(a, np.ndarray) for a in arrays])
        assert all([a.ndim == 2 for a in arrays])

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays))
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        line_kwargs = {k: v for k, v in kwargs.items()
                       if k in self._line_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._line_kwargs}

        # Build the figure and start plotting
        fig = figure if figure else go.Figure()
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
            x = time if time is not None else np.arange(arr.shape[1])

            lines = []
            _legend = True
            for a, y in enumerate(arr):
                if a: _legend = False
                line_kwargs.update({'color': next(colors)})

                lines.append(
                    go.Scatter(x=x, y=y, legendgroup=key, name=key,
                               showlegend=_legend, mode='lines',
                               line=line_kwargs, *args, **kwargs)
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
        fig.update_layout(template=self._template)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_yaxes(**self._default_axis_layout)
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
                 figure: go.Figure = None,
                 title: str = None,
                 x_label: str = None,
                 y_label: str = None,
                 x_limit: Tuple[float] = None,
                 y_limit: Tuple[float] = None,
                 *args, **kwargs
                 ) -> Union[go.Figure, go.FigureWidget]:
        """"""
        # Format data
        assert all([isinstance(a, np.ndarray) for a in arrays])
        assert orientation in ('v', 'h', 'horizontal', 'vertical')

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays))
        if estimator: estimator = self._build_estimator_func(estimator)
        if err_estimator: err_estimator = self._build_estimator_func(err_estimator)
        bar_kwargs = {k: v for k, v in kwargs.items()
                      if k in self._bar_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._bar_kwargs}

        fig = figure if figure else go.Figure()
        for idx, (arr, key) in enumerate(itertools.zip_longest(arrays, keys)):
            # Get the key
            if not key:
                key = f'bar_{idx}'
            key += f' | n={arr.shape[0]}'

            # err_estimator is used to calculate errorbars
            if err_estimator:
                err_arr = err_estimator(arr)

                # If one dimensional, it's the error relative to the mean
                # that's how plotly wants it
                # if if it is 2D, need to subtract from the mean
            else:
                err_arr = None

            # estimator is used to condense all the lines to a single line
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
                        # Assume that they are already set based on the mean value, so that needs
                        # to be subtracted
                        err_plus = arr - err_arr[0, :]
                        err_minus = err_arr[-1, :] - arr
                        error_y.update({'array': err_plus, 'arrayminus': err_minus})
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
                        # Assume that they are already set based on the mean value, so that needs
                        # to be subtracted
                        err_plus = arr - err_arr[0, :]
                        err_minus = err_arr[-1, :] - arr
                        error_x.update({'array': err_plus, 'arrayminus': err_minus})

            # Set up the colors
            _c = next(colors)
            bar_kwargs.update({'marker_color': _c})

            trace = go.Bar(x=x, y=y, error_x=error_x, error_y=error_y,
                           name=key, **bar_kwargs)
            fig.add_trace(trace)

        # Format plot on the way out
        fig.update_traces(**kwargs)
        fig.update_layout(template=self._template, barmode=barmode)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_yaxes(title=y_label, range=y_limit)

        return fig

    def violin_plot(self,
                    arrays: Collection[np.ndarray],
                    neg_arrays: Collection[np.ndarray] = [],
                    keys: Collection[str] = [],
                    neg_keys: Collection[str] = [],
                    colors: Union[str, Collection[str]] = None,
                    neg_colors: Union[str, Collection[str]] = None,
                    orientation: str = 'v',
                    show_box: bool = False,
                    show_points: Union[str, bool] = False,
                    spanmode: str = 'hard',
                    legend: bool = True,
                    figure: go.Figure = None,
                    title: str = None,
                    x_label: str = None,
                    y_label: str = None,
                    x_limit: Tuple[float] = None,
                    y_limit: Tuple[float] = None,
                    *args, **kwargs
                    ) -> Union[go.Figure, go.FigureWidget]:
        """"""
        # Format inputs
        violinmode = None
        side = None
        assert all([isinstance(a, np.ndarray) for a in arrays])
        arrays = [np.squeeze(a) for a in arrays]
        assert all([a.ndim == 1 for a in arrays])
        if neg_arrays:
            assert len(arrays) == len(neg_arrays)
            neg_arrays = [np.squeeze(a) for a in neg_arrays]
            assert all([a.ndim == 1 for a in neg_arrays])
            violinmode = 'overlay'
            side = 'positive'
        assert orientation in ('v', 'h', 'horizontal', 'vertical')

        # Convert any inputs that need converting
        colors = self._build_colormap(colors, len(arrays))
        neg_colors = self._build_colormap(neg_colors, len(neg_arrays))
        violin_kwargs = {k: v for k, v in kwargs.items()
                         if k in self._violin_kwargs}
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._violin_kwargs}

        fig = figure if figure else go.Figure()
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
                              line=line, *args, **violin_kwargs)
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
                                      line=line, *args, **violin_kwargs)
                fig.add_trace(neg_trace)

        # Format plot on the way out
        fig.update_traces(**kwargs)
        fig.update_layout(template=self._template, violinmode=violinmode)
        fig.update_xaxes(**self._default_axis_layout)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._default_axis_layout)
        fig.update_yaxes(title=y_label, range=y_limit)

        return fig

    def heatmap_plot(self,
                     array: np.ndarray,
                     colorscale: str = 'viridis',
                     zmin: float = None,
                     zmid: float = None,
                     zmax: float = None,
                     reverse: bool = False,
                     figure: go.Figure = None,
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     x_limit: Tuple[float] = None,
                     y_limit: Tuple[float] = None,
                     *args, **kwargs
                     ) -> Union[go.Figure, go.FigureWidget]:
        """
        TODO:
            - Add setting the xscale here
        """
        assert array.ndim == 2
        fig = figure if figure else go.Figure()
        trace = go.Heatmap(z=array, zmin=zmin, zmax=zmax,
                           zmid=zmid, colorscale=colorscale,
                           reversescale=reverse,
                           *args, **kwargs)
        fig.add_trace(trace)

        fig.update_layout(template=self._template)
        fig.update_xaxes(**self._no_line_axis)
        fig.update_xaxes(title=x_label, range=x_limit)
        fig.update_yaxes(**self._no_line_axis)
        fig.update_yaxes(title=y_label, range=y_limit)

        return fig









