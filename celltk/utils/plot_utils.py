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

"""
Need some notes for myself, because this is still very hackish, but I
don't have a good idea of how I'd rather set it up. I do know that I want
to be able to make a variety of plots. And instead of passing errors and arrs
around everywhere, they probably only apply to certain plots...

Plotting functions I want:
    - lines
    - lines + shaded area (e.g. sns.tsplot)
    - lines + shaded lines (e.g. eidos model traces)
    - scatter
    - scatter cluster
    - distributions


Need easier way to specify:
    = line colors
    - conditions (maybe in array can have conditions_2_keys or something??)

"""
def plot_groups(arrs: Collection[np.ndarray],
                keys: Collection[str] = [],
                estimator: Union[Callable, str, functools.partial] = None,
                err_estimator: Union[Callable, str, functools.partial] = None,
                colors: (str, Collection) = [],
                kind: str = 'line',
                time: np.ndarray = None,
                legend: bool = True,
                template: str = None,
                figure_spec: dict = {},
                figure: go.Figure = None,
                *args, **kwargs
                ) -> go.Figure:
    """
    Wrapper for plotly functions

    Assume each row in arr is a sample

    TODO:
        - This is currently set up a bit more like a private function
    """
    # Get the plotting function
    try:
        plot_func = PLT_FUNCS[kind]
    except KeyError:
        raise KeyError(f'Could not find function {kind}.')

    # Get the color scale to use
    # colorscale = getattr(pcol.qualitative, DEF_COLORS)
    colorscale = itertools.cycle((cc.glasbey_dark))
    if colors:
        if isinstance(colors, str):
            # TODO: This won't work well for discrete colorscales
            # colorscale = itertools.cycle(pcol.get_colorscale(colors))
            colorscale = itertools.cycle(sns.color_palette(colors, len(arrs)))
        elif isinstance(colors, (list, tuple)):
            colorscale  = itertools.cycle(colors)


    # Make the plot and add the data
    fig = figure if figure else go.Figure()
    for idx, data in enumerate(itertools.zip_longest(arrs, keys)):
        arr, key = data

        # Add the number of cells used to the legend label
        if not key:
            key = idx
        key += f' | n={arr.shape[0]}'

        # Need to calculate error before estimating arr
        if err_estimator:
            err_arr = _apply_estimator(arr, err_estimator)
        else:
            err_arr = None

        arr = _apply_estimator(arr, estimator)

        # Add information to figure
        kwargs.update({'color': _format_colors(next(colorscale))})
        fig = plot_func(fig, arr, err_arr, key, time, *args, **kwargs)

    # Update figure after it is made
    template = template if template else DEF_TEMPLATE
    fig.update_layout(showlegend=legend, template=template, **figure_spec)

    return fig


def get_timeseries_estimator(func: Union[Callable, str],
                             *args, **kwargs
                             ) -> functools.partial:
    """Returns Callable object that will be applied over time axis

    Args:
        func: The function to make the estimator from. If str, looks
            for the function in numpy. Otherwise, uses np.apply_along_axis.

    Returns:
        functools.partial object of the estimator

    TODO:
        - Should have the option to look for already made estimators
    """
    # Remove axis kwarg from kwargs
    kwargs = {k: v for k, v in kwargs.items() if k != 'axis'}

    if isinstance(func, str):
        try:
            func = getattr(celltk.utils.estimator_utils, func)
            return functools.partial(func, *args, **kwargs)
        except AttributeError:
            try:
                func = getattr(np, func)
                return functools.partial(func, axis=0, *args, **kwargs)
            except AttributeError:
                raise ValueError(f'Did not understand estimator {func}')

    else:
        # Need to pass positionally to make it easier to call later
        return functools.partial(np.apply_along_axis, func, 0,
                                 *args, **kwargs)


def plot_trace_predictions(traces: np.ndarray,
                           predictions: np.ndarray,
                           roi: Union[int, Tuple[int]] = None,
                           cmap: str = 'plasma',
                           color_limit: Tuple[int] = (0, 1),
                           y_limit: Tuple[int] = (0, 6),
                           ) -> plt.Figure:
    """"""
    rows, cols = (8, 4)
    size = (11.69, 8.27)  # inches, sheet of paper
    num_figs = int(np.ceil(traces.shape[0] / (rows * cols)))

    # Sum the prediction values
    if roi and predictions.ndim == 3:
        predictions = predictions[..., roi]
    elif predictions.ndim == 3:
        predictions = np.nansum(predictions, axis=-1)

    # Iterate over all the needed figures
    trace_idx = 0

    for fidx in range(num_figs):
        fig, ax = plt.subplots(rows, cols, figsize=size,
                               sharey=True, sharex=True)

        # Iterate over all the axes in the figure and plot
        for a in ax.flatten():
            try:
                line = _make_single_line_collection(traces[trace_idx],
                                                    predictions[trace_idx],
                                                    cmap, color_limit)
                line.set_linewidth(2)
                line = a.add_collection(line)
                trace_idx += 1
            except IndexError:
                # All of the traces have been used
                break

        # Set up the plots
        plt.setp(ax, xlim=(0, traces.shape[1]), ylim=y_limit)
        fig.colorbar(line, ax=ax)

        yield fig


def _apply_estimator(arr: np.ndarray,
                     estimator: Union[Callable, str, functools.partial] = None,
                     ) -> np.ndarray:
    """"""
    if estimator:
        arr = arr.copy()
        if isinstance(estimator, functools.partial):
            # Assume that the estimator is ready to go
            arr = estimator(arr)
        elif isinstance(estimator, (Callable, str)):
            estimator = get_timeseries_estimator(estimator)
            arr = estimator(arr)
        else:
            raise TypeError(f'Did not understand estimator {estimator}')

    return arr


def _line_plot(fig: go.Figure,
               arr: np.ndarray,
               err_arr: np.ndarray = None,
               label: str = None,
               time: np.ndarray = None,
               *args, **kwargs
               ) -> go.Figure:
    """Wrapper for go.Scatter

    https://plotly.com/python/continuous-error-bars/
    """
    kwargs, line_kwargs = _parse_kwargs_for_plot_type('line', kwargs)
    # line_kwargs['color'] = _fmt_
    # TODO: Not sure how to handle this yet...
    x = time if time is not None else np.arange(arr.shape[0])
    # plots one at a time, so arr must be 2D
    if arr.ndim == 1: arr = np.expand_dims(arr, 0)

    # Set up plot
    lines = []
    showlegend = True
    for idx, y in enumerate(arr):
        # Legend is only shown for the first line
        if idx:
            showlegend = False

        lines.append(
            go.Scatter(x=x, y=y, legendgroup=label,
                       showlegend=showlegend, mode='lines',
                       line=line_kwargs,
                       name=label, *args, **kwargs)
        )

        if err_arr is not None:
            # TODO: Are there other forms err_arr could take?
            if err_arr.ndim == 1:
                # Assume it's both high and low
                hi = np.nansum([y, err_arr], axis=0)
                lo = np.nansum([y, -err_arr], axis=0)
            else:
                lo = err_arr[0, :]
                hi = err_arr[-1, :]

            lines.append(
                go.Scatter(x=np.hstack([x, x[::-1]]),
                           y=np.hstack([hi, lo[::-1]]), fill='tozerox',
                           fillcolor=_format_colors(line_kwargs['color'], 0.25),
                           showlegend=False, legendgroup=label,
                           name=label, line=dict(color='rgba(255,255,255,0)'),
                           hoverinfo='skip')
            )

    fig.add_traces(lines)

    return fig


def _overlay_line_plot(fig: go.Figure,
                       arr: np.ndarray,
                       err_arr: np.ndarray,
                       label: str = None,
                       time: np.ndarray = None,
                       *args, **kwargs
                       ) -> go.Figure:
    """"""
    kwargs, line_kwargs = _parse_kwargs_for_plot_type('line', kwargs)

    # TODO: Not sure how to handle this yet...
    x = time if time is not None else np.arange(arr.shape[0])
    # plots one at a time, so arr must be 2D
    if arr.ndim == 1: arr = np.expand_dims(arr, 0)

    # Plot the overlay lines
    lines = []
    # TODO: Check shape of arr here - probably look for order of magnitude
    # alpha = 10. / arr.shape[0]
    alpha = 0.1
    for idx, y in enumerate(arr):
        # Legend is not shown for any of these lines

        lines.append(
            go.Scatter(x=x, y=y, legendgroup=label,
                       showlegend=False, mode='lines',
                       line=dict(color=_format_colors(line_kwargs['color'], alpha)),
                       name=label, *args, **kwargs)
        )

    # Plot the median line
    line_kwargs.update({'width': 5})
    lines.append(
        go.Scatter(x=x, y=np.nanmedian(arr, axis=0),
                   showlegend=True, legendgroup=label, mode='lines',
                   line=line_kwargs, name=label, *args, **kwargs)
    )

    fig.add_traces(lines)

    return fig


def _histogram_plot(fig: go.Figure,
                    arr: np.ndarray,
                    err_arr: np.ndarray,
                    label: str = None,
                    time: np.ndarray = None,
                    *args, **kwargs
                    ) -> go.Figure:
    """"""
    kwargs, hist_kwargs = _parse_kwargs_for_plot_type('hist', kwargs)
    data = go.Histogram(x=np.ravel(arr), legendgroup=label, name=label, showlegend=True,
                   marker=dict(color=_format_colors(hist_kwargs['color'], 0.8)),
                   nbinsx=100, *args, **kwargs, histnorm='percent')
    fig.add_traces(data)

    fig.update_layout(barmode='group')
    fig.update_traces(xbins=dict(start=err_arr.min(), end=err_arr.max()))

    return fig


def _peak_prediction_plot(fig: go.Figure,
                          arr: np.ndarray,
                          err_arr: np.ndarray = None,
                          label: str = None,
                          time: np.ndarray = None,
                          *args, **kwargs
                          ) -> go.Figure:
    """"""


def _color_generator(colors) -> Generator:
    """"""
    pass


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


def _parse_kwargs_for_plot_type(func: str, kwargs: dict) -> dict:
    """Parses kwargs to include the ones that are specific to
    a particular type of plot."""
    plotly_funcs = dict(line=go.Scatter, hist=go.Histogram)
    argnames = inspect.getargspec(plotly_funcs[func]).args

    kept = {}
    other = {}
    for k, v in kwargs.items():
        if k in argnames:
            kept[k] = v
        else:
            other[k] = v

    return kept, other


def _make_single_line_collection(trace: np.ndarray,
                                 prediction: np.ndarray,
                                 cmap: str = 'plasma',
                                 limit: Tuple[int] = (0, 1)
                                 ) -> pltcol.LineCollection:
    """"""
    # Should only get one trace and one probability
    assert trace.ndim == 1
    assert prediction.ndim == 1
    assert len(trace) == len(prediction)

    # Make an array of (ax0, ax1) points
    points = np.array([np.arange(len(trace)), trace]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Set up colors
    norm = plt.Normalize(*limit)
    line = pltcol.LineCollection(segments, cmap=cmap, norm=norm)

    # Add the prediction data and return
    line.set_array(prediction)
    return line


PLT_FUNCS = dict(line=_line_plot, overlay=_overlay_line_plot, hist=_histogram_plot, peaks=_peak_prediction_plot)
DEF_COLORS = 'Safe'
DEF_TEMPLATE = 'simple_white'
