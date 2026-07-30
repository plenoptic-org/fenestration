"""Helpful utility functions for plotting or displaying information."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = []


def __dir__() -> list[str]:
    return __all__


def _setup_fig(image_size: tuple[int, int]) -> Figure:
    """Set up figure for displaying pooling windows.

    Plots an array of ones in the specified size using ``gray_r`` colormap.
    This initializes a figure with a solid black image, removing all x and y ticks.

    Arguments
    ---------
    image_size
        the size of the image to plot. Input should be 2-tuple of ints giving the
        size of the figure in pixels.

    Returns
    -------
    fig
        matplotlib figure containing the plotted images

    """
    # this is an arbitrary value
    ppi = 96

    # get the figure and axes created
    fig = plt.figure(figsize=(image_size[1] / ppi, image_size[0] / ppi), dpi=ppi)

    # define axis parameters
    fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    axes = fig.axes

    # set axes parameters
    axes[0].imshow(np.ones(image_size), cmap="gray_r", interpolation="none")

    return fig
