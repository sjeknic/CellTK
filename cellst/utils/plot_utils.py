import itertools
import functools
import inspect
from typing import Collection, Union, Callable, Generator, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol
import plotly.graph_objects as go
import plotly.colors as pcol
import colorcet as cc

import cellst.utils.estimator_utils


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
                colors: (str, Collection) = [],
                err_arrs: Collection[np.ndarray] = [],
                kind: str = 'line',
                time: np.ndarray = None,
                legend: bool = True,
                template: str = None,
                figure_spec: dict = {},
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
    colorscale = list(cc.glasbey_dark)
    if colors:
        if isinstance(colors, str):
            colorscale = pcol.get_colorscale(colors)
        elif isinstance(colors, (list, tuple)):
            # Make sure enough colors were passed
            if len(colors) >= len(arrs):
                colorscale = colors

    # Make the plot and add the data
    fig = go.Figure()
    for idx, data in enumerate(itertools.zip_longest(arrs, err_arrs, keys)):
        arr, err_arr, key = data
        if not key:
            key = idx

        # Add information to figure
        kwargs.update({'color': colorscale[idx]})
        fig = plot_func(fig, arr, err_arr, key, time, *args, **kwargs)

    # Update figure after it is made
    template = template if template else DEF_TEMPLATE
    fig.update_layout(showlegend=legend, template=template) #, **figure_spec)

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
            func = getattr(cellst.utils.estimator_utils, func)
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
                           save_path: str = 'output'
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

        plt.show()
        plt.close()


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
                           fillcolor=_hex_to_rgba(line_kwargs['color'], 0.25),
                           showlegend=False, legendgroup=label,
                           name=label, line=dict(color='rgba(255,255,255,0)'),
                           hoverinfo='skip')
            )

    fig.add_traces(lines)

    return fig


def _color_generator(colors) -> Generator:
    """"""
    pass


def _hex_to_rgba(color, alpha=1.0) -> Tuple:
    """Converst hexcode colors to RGBA to allow transparency"""
    color = color.lstrip('#')
    col_rgba = list(int(color[i:i+2], 16) for i in (0, 2, 4))
    col_rgba.extend([alpha])
    return 'rgba' + str(tuple(col_rgba))


def _parse_kwargs_for_plot_type(func: str, kwargs: dict) -> dict:
    """Parses kwargs to include the ones that are specific to
    a particular type of plot."""
    plotly_funcs = dict(line=go.Scatter)
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


PLT_FUNCS = dict(line=_line_plot)
DEF_COLORS = 'Safe'
DEF_TEMPLATE = 'simple_white'
