import itertools
import inspect
from typing import Collection

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as pcol
import plotly.io as pio


def plot_groups(arrs: Collection[np.ndarray],
                keys: Collection[str] = [],
                colors: (str, Collection) = [],
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
    """
    # Get the plotting function
    try:
        plot_func = PLT_FUNCS[kind]
    except KeyError:
        raise KeyError(f'Could not find function {kind}.')

    # Get the color scale to use
    colorscale = getattr(pcol.qualitative, DEF_COLORS)
    if colors:
        if isinstance(colors, str):
            colorscale = pcol.get_colorscale(colors)
        elif isinstance(colors, (list, tuple)):
            # Make sure enough colors were passed
            if len(colors) >= len(arrs):
                colorscale = colors

    # Make the plot and add the data
    fig = go.Figure()
    for idx, (arr, key) in enumerate(itertools.zip_longest(arrs, keys)):
        if not key:
            key = idx

        # Add information to figure
        kwargs.update({'color': colorscale[idx]})
        fig = plot_func(fig, arr, key, time, *args, **kwargs)

    template = template if template else DEF_TEMPLATE
    fig.update_layout(showlegend=legend, template=template, **figure_spec)

    return fig


def _line_plot(fig: go.Figure,
               arr: np.ndarray,
               label: str = None,
               time: np.ndarray = None,
               *args, **kwargs
               ) -> go.Figure:
    """Wrapper for go.Scatter"""
    kwargs, line_kwargs = _parse_kwargs_for_plot_type('line', kwargs)

    # Not sure how to handle this yet...
    x = time if time is not None else np.arange(arr.shape[0])
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

    fig.add_traces(lines)

    return fig


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
