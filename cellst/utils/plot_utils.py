import itertools
import functools
import inspect
from typing import Collection, Union, Callable, Generator, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as pcol
import plotly.io as pio
import colorcet as cc

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
    """
    # Remove axis kwarg from kwargs
    kwargs = {k: v for k, v in kwargs.items() if k != 'axis'}

    if isinstance(func, str):
        func = getattr(np, func)
        return functools.partial(func, axis=0, *args, **kwargs)
    else:
        # Need to pass positionally to make it easier to call later
        return functools.partial(np.apply_along_axis, func, 0,
                                 *args, **kwargs)


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
                hi = y + err_arr
                lo = y - err_arr
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


PLT_FUNCS = dict(line=_line_plot)
DEF_COLORS = 'Safe'
DEF_TEMPLATE = 'simple_white'
