"""External utility functions to assist with tensor generation.

pooling_windows.py contains the PoolingWindows class, which uses most of these
functions

"""

from contextlib import suppress

import numpy as np
import torch

__all__ = []


def __dir__() -> list[str]:
    return __all__


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    r"""Cast tensor to numpy in the most conservative way possible.

    Parameters
    ----------
    x
       Tensor to be converted to `np.ndarray` on CPU. If it's already an array,
       we do nothing. We also cast it to float32.

    Returns
    -------
    x
       array version of `x`

    """
    with suppress(AttributeError):
        x = x.detach().cpu().numpy().astype(np.float32)

    return x


def _polar_radius(
    size: int | tuple[int, int],
    exponent: float = 1,
    origin: int | tuple[int, int] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Make distance-from-origin (r) matrix.

    Compute a matrix of given size containing samples of a radial ramp
    function, raised to given exponent, centered at given origin.

    Arguments
    ---------
    size
        if an int, we assume the image should be of dimensions `(size,
        size)`. if a tuple, must be a 2-tuple of ints specifying the
        dimensions
    exponent
        the exponent of the radial ramp function.
    origin
        the center of the image. if an int, we assume the origin is at
        `(origin, origin)`. if a tuple, must be a 2-tuple of ints
        specifying the origin (where `(0, 0)` is the upper left).  if
        None, we assume the origin lies at the center of the matrix,
        `(size+1)/2`.
    device
        the device to create this tensor on

    Returns
    -------
    res
        the polar radius matrix

    """
    if not hasattr(size, "__iter__"):
        size = (size, size)

    if origin is None:
        origin = ((size[0] + 1) / 2.0, (size[1] + 1) / 2.0)
    elif not hasattr(origin, "__iter__"):
        origin = (origin, origin)

    # for some reason, torch.meshgrid returns them in the opposite order
    # that np.meshgrid does. So, in order to get the same output, we
    # grab them as (yramp, xramp) instead of (xramp, yramp). similarly,
    # we have to reverse the order from (size[1], size[0]) to (size[0],
    # size[1])
    yramp, xramp = torch.meshgrid(
        torch.arange(1, size[0] + 1, device=device) - origin[0],
        torch.arange(1, size[1] + 1, device=device) - origin[1],
    )

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp**2 + yramp**2
        res = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        res = (xramp**2 + yramp**2) ** (exponent / 2.0)
    return res


def _polar_angle(
    size: int | tuple,
    phase: float = 0,
    origin: int | tuple[int, int] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Make polar angle matrix (in radians).

    Compute a matrix of given size containing samples of the polar angle (in radians,
    CW from the X-axis, ranging from -pi to pi), relative to given phase, about the
    given origin pixel.

    Arguments
    ---------
    size
        if an int, we assume the image should be of dimensions `(size, size)`. if a
        tuple, must be a 2-tuple of ints specifying the dimensions
    phase
        the phase of the polar angle function (in radians, clockwise from the X-axis)
    origin
        the center of the image. if an int, we assume the origin is at
        `(origin, origin)`. if a tuple, must be a 2-tuple of ints specifying the
        origin (where `(0, 0)` is the upper left). if None, we assume the origin lies
        at the center of the matrix, `(size+1)/2`.
    device
        the device to create this tensor on

    Returns
    -------
    res
        the polar angle matrix

    """
    if not hasattr(size, "__iter__"):
        size = (size, size)

    if origin is None:
        origin = ((size[0] + 1) / 2.0, (size[1] + 1) / 2.0)
    elif not hasattr(origin, "__iter__"):
        origin = (origin, origin)

    # for some reason, torch.meshgrid returns them in the opposite order
    # that np.meshgrid does. So, in order to get the same output, we
    # grab them as (yramp, xramp) instead of (xramp, yramp). similarly,
    # we have to reverse the order from (size[1], size[0]) to (size[0],
    # size[1])
    yramp, xramp = torch.meshgrid(
        torch.arange(1, size[0] + 1, device=device) - origin[0],
        torch.arange(1, size[1] + 1, device=device) - origin[1],
    )

    res = torch.atan2(yramp, xramp)

    res = ((res + (np.pi - phase)) % (2 * np.pi)) - np.pi

    return res
