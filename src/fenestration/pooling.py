"""Functions to perform spatial pooling, as seen in Freeman and Simoncelli, 2011.

In addition the raised-cosine windows used in that paper, we also
provide support for an alternative window construction:
Gaussians. They're laid out in the same fashion as the raised-cosine
windows, but are wider and have values everywhere (whereas the
raised-cosine windows are clipped so that they're zero for most of the
image). Using the raised-cosine windows led to issues with aliasing in
metamer synthesis, visible as ringing, with the PooledV1
model, because of the interactions between the windows and the steerable
pyramid filters.

The Gaussian windows don't have these problems, but require more windows to
evenly tile the image in the radial direction (and thus PoolingWindows.forward
will take more memory and more time). Note as well that, whereas the max
amplitude of the raised-cosine windows is always 1 (for all transition region
widths), the Gaussian windows will have their max amplitude scaled down as
their standard deviation increases; as the standard deviation increases, the
windows overlap more, so that the number of windows a given pixel lies in
increases and thus the weighting in each of them needs to decrease in order to
make sure the sum across all windows is still 1 for every pixel. The Gaussian
windows will always intersect at x=.5, but the interpretation of this depends
on its standard deviation. For Gaussian windows, we recommend (and only
support) a standard deviation of 1, so that each window intersects at half a
standard deviation.

pooling_windows.py contains the PoolingWindows class, which uses most of these
functions

"""

import math
import warnings
from typing import Literal

import numpy as np
import torch

from . import _tensors, calculate

__all__ = [
    "gaussian",
    "raised_cosine",
    "normalize_windows",
]


def __dir__() -> list[str]:
    return __all__


# see docstring of gaussian function for explanation of this constant
GAUSSIAN_SUM = 2 * 1.753314144021452772415339526931980189073725635759454989253 - 1


def gaussian(x: float | np.ndarray, std_dev: float | None = 1) -> np.ndarray:
    r"""Compute simple gaussian with mean 0, and adjustable std dev.

    Possible window function, giving the weighting in each
    direction for the spatial pooling performed during the construction
    of visual metamers.

    Parameters
    ----------
    x
        The distance in a direction
    std_dev
        The standard deviation of the Gaussian window.

    Returns
    -------
    array
        The value of the window at each value of `x`

    Notes
    -----
    We normalize in here in order to make sure that the windows sum to
    1. In order to do that, we note that each Gaussian is centered at
    integer x values: 0, 1, 2, 3, etc. If we're summing at ``x=0``, we
    then note that the first window will be centered there and so have
    its max amplitude, its two nearest neighbors will be 1 away from
    their center (these Gaussians are symmetric), their two nearest
    neighbors will be 2 away from their center, etc. Therefore, we'll
    have one Gaussian at max value (1), two at
    :math:`\exp(\frac{-1^2}{2\sigma^2})`, two at
    :math:`\exp(\frac{-2^2}{2\sigma^2})`, etc.

    Summing at this location will give us the value we need to normalize
    by, :math:`S`. We work through this with :math:`\sigma=1`:

    .. math::

        S &= 1 + 2 * \exp(\frac{-(1)^2}{2\sigma^2}) + 2 * \exp(\frac{-(2)^2}{2\sigma^2})
         + ... \\
        S &= 1 + 2 * \sum_{n=1}^{\inf} \exp(\frac{-n^2}{2}) \\
        S &= -1 + 2 * \sum_{n=0}^{\inf} \exp(\frac{-n^2}{2})

    And we've stored this number as the constant ``GAUSSIAN_SUM`` (the
    infinite sum computed in the equation above was using
    `Wolfram Alpha <https://www.wolframalpha.com/input/?i=sum+0+to+inf+e%5E%28-n%5E2%2F2%29+>`_.

    When ``std_dev>1``, the windows overlap more. As with the
    probability density function of a normal distribution, we divide by
    :attr:`std_dev` to keep the integral constant for different values of
    :attr:`std_dev` (though the integral is not 1). This means that summing
    across multiple windows will still give us a value of 1.

    """
    return torch.exp(-(x**2 / (2 * std_dev**2))) / (std_dev * GAUSSIAN_SUM)


def raised_cosine(
    x: float | np.ndarray, transition_region_width: float = 0.5
) -> np.ndarray:
    r"""Compute raised cosine window function.

    Possible window function, giving the weighting in each
    direction for the spatial pooling performed during the construction
    of visual metamers.

    Parameters
    ----------
    x
        The distance in a direction
    transition_region_width
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. Must lie between 0 and 1.

    Returns
    -------
    array
        The value of the raised cosine window at each value of :attr:`x`.

    Raises
    ------
    Exception
        If :attr:`transition_region_width` is not between 0 and 1

    Notes
    -----
    For :attr:`x` values outside the function's domain, we return 0

    Equation 9 from the online methods of [3]_.

    References
    ----------
    .. [3] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream.
        Nature Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if transition_region_width > 1 or transition_region_width < 0:
        raise Exception("transition_region_width must lie between 0 and 1!")
    # doing it in this array-ized fashion is much faster
    y = torch.zeros_like(x)
    # this creates a bunch of masks
    masks = [
        (-(1 + transition_region_width) / 2 < x)
        & (x <= (transition_region_width - 1) / 2),
        ((transition_region_width - 1) / 2 < x)
        & (x <= (1 - transition_region_width) / 2),
        ((1 - transition_region_width) / 2 < x)
        & (x <= (1 + transition_region_width) / 2),
    ]
    # and this creates the values where those masks are
    vals = [
        torch.cos(
            np.pi
            / 2
            * ((x - (transition_region_width - 1) / 2) / transition_region_width)
        )
        ** 2,
        torch.ones_like(x),
        (
            -(
                torch.cos(
                    np.pi
                    / 2
                    * (
                        (x - (1 + transition_region_width) / 2)
                        / transition_region_width
                    )
                )
                ** 2
            )
            + 1
        ),
    ]
    for m, v in zip(masks, vals):
        y[m] = v[m]
    return y


def _polar_angle_windows(
    n_windows: int,
    resolution: int | tuple[int, int],
    window_type: Literal["cosine", "gaussian"] = "gaussian",
    transition_region_width: float | None = None,
    std_dev: float | None = 1,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    r"""Create polar angle windows.

    We require an integer number of windows placed between 0 and :math:`2 \pi`.

    Parameters
    ----------
    n_windows
        The number of polar angle windows we create.
    resolution
        2-tuple of ints specifying the resolution of the 2d images to
        make. If an int, will only make the windows in 1d (this is
        mainly for testing purposes)
    window_type
        Whether to use the raised cosine function from [4]_ or a
        Gaussian that has approximately the same structure. If cosine,
        :attr:`transition_region_width` must be set; if gaussian, then
        :attr:`std_dev` must be set
    transition_region_width
        The width of the cosine windows' transition region, parameter
        :math:`t` in equation 9 from the online methods.
    std_dev
        The standard deviation of the Gaussian window.
    device
        the device to create this tensor on

    Returns
    -------
    windows
        A 3d tensor containing the (2d) polar angle windows. Windows
        will be indexed along the first dimension. If resolution was an
        int, then this will be a 2d arra containing the 1d polar angle
        windows

    Raises
    ------
    Exception
        If :attr:`n_windows` is not an integer
    Exception
        If :attr:`n_windows` is not greater than 8*``std_dev``

    Notes
    -----
    Equation 10 from the online methods of [4]_.

    References
    ----------
    .. [4] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    if int(n_windows) != n_windows:
        raise Exception("n_windows must be an integer!")
    if n_windows == 1:
        raise Exception("We cannot handle one window correctly!")
    # this is `w_\theta` in the paper
    window_spacing = calculate._angular_window_spacing(n_windows)
    max_angle = 2 * np.pi - window_spacing
    if window_type == "gaussian" and (std_dev * 8) > n_windows:
        raise Exception(
            f"In order for windows to tile the circle correctly, n_windows ({n_windows}"
            f") must be greater than 8*std_dev ({8 * std_dev})!"
        )
    if hasattr(resolution, "__iter__") and len(resolution) == 2:
        theta = _tensors._polar_angle(resolution, device=device).unsqueeze(0)
        theta = theta + (
            np.pi
            - torch.linspace(0, max_angle, n_windows, device=device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
    else:
        theta = torch.linspace(0, 2 * np.pi, resolution, device=device).unsqueeze(0)
        theta = theta + (
            np.pi - torch.linspace(0, max_angle, n_windows, device=device).unsqueeze(-1)
        )
    theta = ((theta % (2 * np.pi)) - np.pi) / window_spacing
    if window_type == "gaussian":
        windows = gaussian(theta, std_dev)
    elif window_type == "cosine":
        windows = raised_cosine(theta, transition_region_width)
    return torch.stack([w for w in windows if (w != 0).any()])


def _log_eccentricity_windows(
    resolution: int | tuple[int, int],
    n_windows: float | None = None,
    window_spacing: float | None = None,
    min_ecc: float = 0.5,
    max_ecc: float = 15,
    window_type: Literal["cosine", "gaussian"] = "gaussian",
    transition_region_width: float | None = None,
    std_dev: float | None = 1,
    device: str | torch.device | None = None,
    linear: bool = False,
) -> torch.Tensor:
    r"""Create log eccentricity windows in 2d.

    Note that exactly one of ``n_windows`` or ``window_width`` must be
    set.

    In order to convert the polar radius array we create from pixels to
    degrees, we assume that ``max_ecc`` is the maximum eccentricity in
    the whichever is the larger dimension (i.e., to convert from pixels
    to degrees, we multiply by ``max_ecc / (max(resolution)/2)``)

    NOTE: if ``n_windows`` (rater than ``window_width``) is set, this is
    not necessarily the number of arrays we'll return. In order to get
    the full set of windows, we want to consider those that would show
    up in the corners as well, so it's probable that this function
    returns one more window there; we determine if this is necessary by
    calling ``calculate._eccentricity_n_windows`` with
    ``np.sqrt(2)*max_ecc``.

    Parameters
    ----------
    resolution
        2-tuple of ints specifying the resolution of the 2d images to
        make. If an int, will only make the windows in 1d (this is
        mainly for testing purposes)
    n_windows
        The number of log-eccentricity windows from ``min_ecc`` to
        ``max_ecc``. ``n_windows`` xor ``window_width`` must be set.
    window_spacing
        The spacing of the log-eccentricity windows. ``n_windows`` xor
        ``window_spacing`` must be set.
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    window_type
        Whether to use the raised cosine function from [5]_ or a
        Gaussian that has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set
    transition_region_width
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    std_dev
        The standard deviation of the Gaussian window. WARNING -- For
        now, we only support ``std_dev=1`` (in order to ensure that the
        windows tile correctly, intersect at the proper point, follow
        scaling, and have proper aspect ratio; not sure we can make that
        happen for other values).
    device
        the device to create this tensor on
    linear
        if True, create linear windows instead of log-spaced. NOTE This is only
        for playing around with, it really is not supported or a good idea
        because the angular windows still grow in size as a function of
        eccentricity and none of the calculations will work.

    Returns
    -------
    windows
        A 3d tensor containing the (2d) log-eccentricity
        windows. Windows will be indexed along the first dimension. If
        resolution was an int, then this will be a 2d array containing
        the 1d polar angle windows

    Raises
    ------
    Warning
        If ``std_dev`` is not 1

    Notes
    -----
    Equation 11 from the online methods of [5]_.

    References
    ----------
    .. [5] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    log_func = torch.log if not linear else lambda x: x
    if std_dev is not None and std_dev != 1:
        warnings.warn(
            "Only std_dev=1 has been tested -- not sure if Gaussian "
            "windows will uniformly tile image otherwise! Use at your own risk."
        )
    if window_spacing is None:
        window_spacing = calculate._eccentricity_window_spacing(
            min_ecc, max_ecc, n_windows, std_dev=std_dev
        )
    n_windows = calculate._eccentricity_n_windows(
        window_spacing, min_ecc, max_ecc * np.sqrt(2), std_dev
    )
    shift_arg = (
        log_func(torch.tensor(min_ecc, dtype=torch.float32))
        + window_spacing * torch.arange(1, math.ceil(n_windows) + 1, device=device)
    ).unsqueeze(-1)
    if hasattr(resolution, "__iter__") and len(resolution) == 2:
        ecc = log_func(
            _tensors._polar_radius(resolution, device=device)
            / calculate.deg_to_pix(resolution, max_ecc)
        ).unsqueeze(0)
        shift_arg = shift_arg.unsqueeze(-1)
    else:
        ecc = log_func(torch.linspace(0, max_ecc, resolution, device=device)).unsqueeze(
            0
        )
    ecc = (ecc - shift_arg) / window_spacing
    if window_type == "gaussian":
        windows = gaussian(ecc, std_dev)
    elif window_type == "cosine":
        windows = raised_cosine(ecc, transition_region_width)
    return torch.stack([w for w in windows if (w != 0).any()])


def create_pooling_windows(
    scaling: float | None,
    img_res: tuple[int, int],
    min_ecc: float = 0.5,
    max_ecc: float = 15,
    radial_to_circumferential_ratio: float = 2,
    window_type: Literal["cosine", "gaussian"] = "gaussian",
    transition_region_width: float | None = None,
    std_dev: float | None = 1,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor | dict, torch.Tensor | dict]:
    r"""
    Create two sets of 2d pooling windows that span the visual field.

    This creates the pooling windows that we use to average image
    statistics for metamer generation as done in [6]_. This is returned
    as two 3d torch tensors for further use with a model.

    Note that these are returned separately as log-eccentricity and
    polar angle tensors and if you want the windows used in the paper
    [6]_, you'll need to call :func:`torch.einsum` (see Examples section)
    or, better yet, use the :class:`PoolingWindows` class, which is provided
    for this purpose.

    Parameters
    ----------
    scaling
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity (see the :meth:`calculate.scaling` function).
    img_res
        2-tuple of ints specifying the resolution of the 2d images to
        make.
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    radial_to_circumferential_ratio
        :attr:`scaling` determines the number of log-eccentricity windows we
        can create; this ratio gives us the number of polar angle
        ones. Based on :attr:`scaling`, we calculate the width of the windows
        in log-eccentricity, and then divide that by this number to get
        their width in polar angle. Because we require an integer number
        of polar angle windows, we round the resulting number of polar
        angle windows to the nearest integer, so the ratio in the
        generated windows approximate this. 2 (the default) is the value
        used in the paper [6]_.
    window_type
        Whether to use the raised cosine function from [6]_ or a Gaussian that
        has approximately the same structure. If cosine,
        :attr:`transition_region_width` must be set; if gaussian, then :attr:`std_dev`
        must be set.
    transition_region_width
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    std_dev
        The standard deviation of the Gaussian window. WARNING -- if
        this is too small (say < 3/4), then the windows won't tile
        correctly. So we only support std_dev=1 for now.
    device
        the device to create these tensors on

    Returns
    -------
    angle_windows
        The 3d tensor of 2d polar angle windows. Its shape will be
        ``(n_angle_windows, *img_res)``, where the number of windows
        is inferred in this function based on the values of :attr:`scaling`
        and :attr:`radial_to_circumferential_width`.
    ecc_windows
        The 3d tensor of 2d log-eccentricity windows. Its shape will be
        ``(n_eccen_windows, *img_res)``, where the number of windows
        is inferred in this function based on the values of :attr:`scaling`,
        :attr:`min_ecc`, and :attr:`max_ecc`.

    See Also
    --------
    PoolingWindows : generate PoolingWindows object

    References
    ----------
    .. [6] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
        ventral stream. Nature Neuroscience, 14(9),
        1195–1201. http://dx.doi.org/10.1038/nn.2889
    .. [7] Broderick, W. F., Rufo, G., Winawer, J. & Simoncelli, E. P.
        (2023). Foveated metamers of the early visual system. eLife,
        12:RP90554. http://dx.doi.org/10.7554/eLife.90554.2

    Examples
    --------
    To use, simply call with the desired scaling and image size (for the
    version seen in the paper, don't change any of the default arguments;
    compare this image to the right side of Supplementary Figure 1C).

    Although we have hard-coded the standard deviation (to 1, for
    ``window_type="gaussian"``) and transition region width (to 0.5, for
    ``window_type="cosine"``) when creating the :class:`PoolingWindows` object, it is
    possible to manually adjust these parameters when using
    :meth:`create_pooling_windows`. However, only the default values have been tested!
    It is unclear whether the windows will uniformly tile the images otherwise.

    To create gaussian windows (default), you can specify the following arguments:

    >>> import fenestration as fen
    >>> angle_w, ecc_w = fen.create_pooling_windows(
    ...     scaling=0.8,
    ...     img_res=(256, 256),
    ...     min_ecc=1,
    ...     max_ecc=10,
    ...     radial_to_circumferential_ratio=2,
    ...     window_type="gaussian",
    ...     transition_region_width=None,
    ...     std_dev=1,
    ...     device="cpu",
    ... )

    Similarly for raised cosine windows:

    >>> angle_w, ecc_w = fen.create_pooling_windows(
    ...     scaling=0.8,
    ...     img_res=(256, 256),
    ...     min_ecc=1,
    ...     max_ecc=10,
    ...     radial_to_circumferential_ratio=2,
    ...     window_type="cosine",
    ...     transition_region_width=0.5,
    ...     std_dev=None,
    ...     device="cpu",
    ... )

    To create equivalent windows to what is generated by :class:`PoolingWindows`,
    you must also normalize resulting windows so they have an L1-norm of 1. This
    is useful when generating model metamers using the windows' representation
    so that each eccentricity contributes equally, facilitating optimization.
    See [7]_ for an example.

    You can display the various angle and eccentricity windows by plotting a
    specified index:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import fenestration as fen
        >>> import matplotlib.pyplot as plt
        >>> angle_w, ecc_w = fen.create_pooling_windows(0.8, (256, 256))
        >>> fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(ecc_w[0], cmap="Grays_r", interpolation="none")
        <matplotlib.image.AxesImage ...>
        >>> ax[1].imshow(angle_w[0], cmap="Grays_r", interpolation="none")
        <matplotlib.image.AxesImage ...>

    If you wish to get the windows as shown in Supplementary Figure 1C
    in the paper [6]_, use :func:`torch.einsum` (if you wish to apply these
    to images, use the :class:`PoolingWindows` class instead, which has many
    more features):

    .. plot::
        :include-source:
        :context: close-figs

        >>> import fenestration as fen
        >>> import torch
        >>> angle_w, ecc_w = fen.create_pooling_windows(0.8, (256, 256))
        >>> # we ignore the last ring of eccentricity windows here because
        >>> # they're all relatively small, which makes the following plot
        >>> # look weird. for how to properly handle them, see the
        >>> # PoolingWindows class
        >>> windows = torch.einsum("ahw,ehw->eahw", [angle_w, ecc_w[:-1]]).flatten(0, 1)
        >>> fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        >>> # we use the intersecting amplitude value for gaussian windows, 0.14
        >>> for w in windows:
        ...     ax.contour(w, [0.14], colors="r")
        <matplotlib.contour.QuadContourSet ...>


    """
    ecc_window_spacing = calculate._eccentricity_window_spacing(
        min_ecc, max_ecc, scaling=scaling, std_dev=std_dev
    )
    n_polar_windows = calculate._angular_n_windows(
        ecc_window_spacing / radial_to_circumferential_ratio
    )
    # we want to set the number of polar windows where the ratio of
    # widths is approximately what the user specified. the constraint
    # that it's an integer is more important
    n_polar_windows = int(round(n_polar_windows))
    angle_tensor = _polar_angle_windows(
        n_polar_windows,
        img_res,
        window_type,
        transition_region_width=transition_region_width,
        std_dev=std_dev,
        device=device,
    )
    ecc_tensor = _log_eccentricity_windows(
        img_res,
        None,
        ecc_window_spacing,
        min_ecc,
        max_ecc,
        window_type,
        std_dev=std_dev,
        transition_region_width=transition_region_width,
        device=device,
    )
    return angle_tensor, ecc_tensor


def normalize_windows(
    angle_windows: torch.Tensor,
    ecc_windows: torch.Tensor,
    window_eccentricity: np.ndarray,
) -> tuple[dict, torch.Tensor]:
    r"""Normalize windows to have L1-norm of 1.

    We calculate the L1-norm of single windows (that is, product of
    eccentricity and angular windows) for all angles, one middling
    eccentricity (third of the way thorugh), then average across angles
    (because of alignment with pixel grid, L1-norm will vary somewhat
    across angles).

    L1-norm scales linearly with area, which is proportional to the width in
    the angular direction times the width in the radial direction. The angular
    width grows linearly with eccentricity, while the radial width grows with
    the reciprocal of the derivative of our scaling function (that's log(ecc)
    for gaussian windows). so we use that product to scale it for the different
    windows. only eccentricity windows is normalized (don't need to divide
    both).

    Parameters
    ----------
    angle_windows
        tensor containing the angular windows
    ecc_windows
        tensor containing the eccentricity windows
    window_eccentricity
        array containing the eccentricity for each window that defines
        their location relative to each other (and so can be in either
        pixels or degrees). this is used to determine how to scale the
        L1-norm. It should probably be the central eccentricity, but it
        should not contain any zeros.

    Returns
    -------
    ecc_windows
        the normalized ecc_windows.
    scale_factor
        the scale_factor used to normalize eccentricity windows
        (as a 3d tensor, number of eccentricity windows by 1 by 1). Stored by
        :class:`~fenestration.PoolingWindows` object so we can undo it for
        :meth:`~fenestration.PoolingWindows.project` or plotting purposes.

    """
    # pick some window with a middling eccentricity
    n = ecc_windows.shape[0] // 3
    # get the l1 norm of a single window
    w = torch.einsum("ahw,hw->ahw", angle_windows, ecc_windows[n])
    l1 = torch.norm(w, 1, (-1, -2))
    l1 = l1.mean(0)
    # the l1 norm grows with the area of the windows; the radial
    # direction width grows with the reciprocal of the derivative of
    # log(ecc), which is ecc, and the angular direction width grows
    # with the eccentricity as well. so l1 norm grows with the
    # eccentricity squared
    deriv = torch.tensor(window_eccentricity**2, dtype=torch.float32)
    deriv_scaled = deriv / deriv[n]
    scale_factor = 1 / (deriv_scaled * l1).to(torch.float32)
    while scale_factor.ndim < 3:
        scale_factor = scale_factor.unsqueeze(-1)
    # there's a chance we'll have more windows accounted for in
    # scale factor then we actually made (because we calculate
    # details for windows that go out farther, just in case). if
    # that's so, drop the extra scale factor
    if len(scale_factor) > len(ecc_windows):
        scale_factor = scale_factor[: len(ecc_windows)]
    ecc_windows = ecc_windows * scale_factor
    return ecc_windows, scale_factor
