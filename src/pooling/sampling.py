"""Functions related to sampling.

handful of functions here, related to sampling and checking whether
you're sampling correctly, in order to avoid aliasing

when doing something like strided convolution or using the pooling windows,
you want to make sure you're sampling the image appropriately, in
order to avoid aliasing. this file contains some functions to help you with
that, see the Sampling_and_Aliasing notebook for some examples

"""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.figure import Figure

from . import tensors
from .pooling import gaussian

__all__ = ["check_sampling", "plot_coeffs", "interpolation_plot", "create_movie"]


def __dir__() -> list[str]:
    return __all__


def check_sampling(
    val_sampling: float | None = 0.5,
    pix_sampling: int | None = None,
    func: Callable[[float | np.ndarray], np.ndarray] = gaussian,
    x: torch.Tensor | np.ndarray = torch.linspace(-5, 5, 101),
    **func_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Check how sampling relates to interpolation quality.

    Given a function, a domain, and how to sample that domain, this
    function will use linear algebra (``np.linalg.lstsq``) to determine
    how to interpolate the function so that it's centered on each
    pixel. You can then use functions like ``plot_coeffs`` and
    ``create_movie`` to see the quality of this interpolation

    The idea here is to take a function (for example,
    ``pooling.pooling.gaussian``) and say that we have this function
    defined at, e.g., every 10 pixels on the array ``linspace(-5, 5,
    101)``. We want to answer then, the question of how well we can
    interpolate to all the intermediate functions, that is, the
    functions centered on each pixel in the array.

    You can either specify the spacing in pixels (``pix_sampling``) xor
    in x values (``val_sampling``), but exactly one of them must be set.

    Your function can either be a torch or numpy function, but ``x``
    must be the appropriate type, we will not cast it for you.

    Parameters
    ----------
    val_sampling
        If float, how far apart (in x-values) each sampled function
        should be. This doesn't have to align perfectly with the pixels,
        but should be close. If None, we use ``pix_sampling`` instead.
    pix_sampling
        If int, how far apart (in pixels) each sampled function should
        be. If None, we use ``val_sampling`` instead.
    func
        the function to check interpolation for. must take ``x`` as its
        first input, all additional kwargs can be specified in
        ``func_kwargs``
    x
        the 1d tensor/array to evaluate ``func`` on.
    func_kwargs
        additional kwargs to pass to ``func``

    Returns
    -------
    sampled
        the array of sampled functions. will have shape ``(len(x),
        ceil(len(x)/pix_sampling))``
    full
        the array of functions centered at each pixel. will have shape
        ``(len(x), len(x))``
    interpolated
        the array of functions interpolated to each pixel. will have
        shape ``(len(x), len(x))``
    coeffs
        the array of coefficients to transform ``sampled`` to
        ``full``. This has been transposed from the array returned by
        ``np.linalg.lstsq`` and thus will have the same shape as
        ``sampled`` (this is to make it easier to restrict which coeffs
        to look at, since they'll be more easily indexed along first
        dimension)
    residuals
        the errors for each interpolation, will have shape ``len(x)``

    Raises
    ------
    Exception
        If neither ``val_sampling`` nor ``pix_sampling`` are set to ``None``

    """
    if val_sampling is not None:
        if pix_sampling is not None:
            raise Exception("One of val_sampling or pix_sampling must be None!")
        # this will get us the closest value, if there's no exactly
        # correct one.
        pix_sampling = np.argmin(abs((x + val_sampling)[0] - x))
        if pix_sampling == 0 or pix_sampling == (len(x) - 1):
            # the above works if x is increasing. if it's decreasing,
            # then pix_sampling will be one of the extremal values, and
            # we need to try the following
            pix_sampling = np.argmin(abs((x - val_sampling)[0] - x))
    try:
        X = x.unsqueeze(1) - x[::pix_sampling]
        sampled = tensors.to_numpy(func(X, **func_kwargs))
        full_X = x.unsqueeze(1) - x
        full = tensors.to_numpy(func(full_X, **func_kwargs))
    except AttributeError:
        # numpy arrays don't have unsqueeze, so we use this `[:, None]`
        # syntax to get the same outcome
        X = x[:, None] - x[::pix_sampling]
        sampled = func(X, **func_kwargs)
        full_X = x.unsqueeze(1) - x
        full = func(full_X, func_kwargs)
    coeffs, residuals, _, _ = np.linalg.lstsq(sampled, full, rcond=None)
    interpolated = np.matmul(sampled, coeffs)
    return sampled, full, interpolated, coeffs.T, residuals


def plot_coeffs(
    coeffs: np.ndarray, ncols: int = 5, ax_size: tuple[int, int] = (5, 5)
) -> Figure:
    r"""Plot interpolation coefficients.

    Simple function to plot a bunch of interpolation coefficients on the
    same figure as stem plots

    Parameters
    ----------
    coeffs
        the array of coefficients to transform ``sampled`` to
        ``full``. In order to show fewer coefficients (because they're
        so many), index along the first dimension (e.g., ``coeffs[:10]``
        to view first 10)
    ncols
        the number of columns to create in the plot
    ax_size
        the size of each subplots axis

    Returns
    -------
    fig
        the figure containing the plot
    """
    nrows = int(np.ceil(coeffs.shape[0] / ncols))
    ylim = max(abs(coeffs.max()), abs(coeffs.min()))
    ylim += ylim / 10
    fig, axes = plt.subplots(
        nrows, ncols, figsize=[i * j for i, j in zip(ax_size, [ncols, nrows])]
    )
    for i, ax in enumerate(axes.flatten()):
        ax.stem(coeffs[i])
        ax.set_ylim((-ylim, ylim))
    return fig


def interpolation_plot(
    interpolated: np.ndarray,
    residuals: np.ndarray,
    pix: int | None = 0,
    val: float | None = None,
    x: torch.Tensor | np.ndarray = np.linspace(-5, 5, 101),
    full: np.ndarray | None = None,
) -> Figure:
    r"""Create plot showing interpolation results at specified pixel or value.

    We have two subplots: the interpolation (with optional actual
    values) and the residuals

    Either ``pix`` or ``val`` must be set, and the other must be
    ``None``. They specify which interpolated function to display

    Parameters
    ----------
    interpolated
        the array of functions interpolated to each pixel
    residuals
        the errors for each interpolation
    pix
        we plot the interpolated function centered at this pixel
    val
        we plot the interpolated function centered at this x-value
    x
        the 1d tensor/array passed to ``check_sampling()``. the default
        here is the default there. plotted on x-axis
    full
        the array of functions centered at each pixel. If None, won't
        plot. If not None, will plot as dashed line behind the
        interpolation for comparison

    Returns
    -------
    fig
        figure containing the plot

    Raises
    ------
    Exception
        If neither ``val_sampling`` nor ``pix_sampling`` are set to ``None``

    """
    if val is not None:
        if pix is not None:
            raise Exception("One of val_sampling or pix_sampling must be None!")
        # this will get us the closest value, if there's no exactly
        # correct one.
        pix = np.argmin(abs(x - val))
    x = tensors.to_numpy(x)
    ylim = [interpolated.min(), interpolated.max()]
    ylim = [ylim[0] - np.diff(ylim) / 10, ylim[1] + np.diff(ylim) / 10]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_ylim(ylim)
    axes[0].plot(x, interpolated[:, pix], zorder=0, label="interpolation")
    if full is not None:
        axes[0].plot(x, full[:, pix], "--", label="actual")
        axes[0].legend()
    axes[1].stem(x, residuals)
    axes[1].scatter(x[pix], residuals[pix], c="r", zorder=10)
    axes[0].set_title("Interpolated function centered at highlighted pixel")
    axes[1].set_title("Error for interpolation centered at highlighted pixel")
    return fig


def create_movie(
    interpolated: np.ndarray,
    residuals: np.ndarray,
    x: torch.Tensor | np.ndarray = np.linspace(-5, 5, 101),
    full: np.ndarray | None = None,
    framerate: int = 10,
) -> animation.FuncAnimation:
    r"""Create movie showing the interpolation results.

    We create a simple movie to show this in action. we have two
    subplots: the interpolation (with optional actual values) and the
    residuals.

    the more finely sampled your ``x`` was when calling
    ``check_sampling()`` (and thus the larger your ``interpolated`` and
    ``full`` arrays), the longer this will take. Calling this function
    will not take too long, but displaying or saving the returned
    animation will.

    Parameters
    ----------
    interpolated
        the array of functions interpolated to each pixel
    residuals
        the errors for each interpolation
    x
        the 1d tensor/array passed to ``check_sampling()``. the default
        here is the default there. plotted on x-axis
    full
        the array of functions centered at each pixel. If None, won't
        plot. If not None, will plot as dashed line behind the
        interpolation for comparison
    framerate
        How many frames a second to display.

    Returns
    -------
    anim
        The animation object. In order to view, must convert to HTML
        (call ``pooling.tensors.convert_anim_to_html(anim)``) or save (call
        ``anim.save(movie.mp4)``, must have ``ffmpeg`` installed).

    """
    x = tensors.to_numpy(x)
    fig = interpolation_plot(interpolated, residuals, x=x, full=full)
    if full is not None:
        full_line = fig.axes[0].lines[1]
    interp_line = fig.axes[0].lines[0]
    scat = fig.axes[1].collections[1]

    def _movie_plot(i):  # noqa: ANN202, ANN001
        interp_line.set_data(x, interpolated[:, i])
        scat.set_offsets((x[i], residuals[i]))
        artists = [interp_line, scat]
        if full is not None:
            full_line.set_data(x, full[:, i])
            artists.append(full_line)
        return artists

    return animation.FuncAnimation(
        fig,
        _movie_plot,
        frames=len(interpolated),
        blit=True,
        interval=1000.0 / framerate,
        repeat=False,
    )
