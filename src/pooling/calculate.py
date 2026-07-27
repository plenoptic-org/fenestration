"""Helpfer functions for calculating pooling window components."""

import re
from typing import Literal

import numpy as np

__all__ = ["scaling", "deg_to_pix"]


def __dir__() -> list[str]:
    return __all__


def _angular_window_spacing(n_windows: float) -> float:
    r"""Calculate and return the window spacing for the angular windows.

    this is the :math:`w_{\theta }` term in equation 10 of the paper's
    online methods, referred to as the angular window width.

    For both cosine and gaussian windows, this is the distance between
    the peaks of the windows. For cosine windows, this is also the same
    as the windows' widths, but gausian windows' widths are
    approximately ``window_spacing * std_dev * 3`` (since they're
    Gaussian, 99.73% of their mass lie within 3 standard deviations, but
    the Gaussians are technically infinite)

    Parameters
    ----------
    n_windows
        The number of windows to pack into 2 pi. Note that we don't
        require it to be an integer here, but the code that makes use of
        this does.

    Returns
    -------
    window_spacing
        The spacing of the polar angle windows.

    """
    return (2 * np.pi) / n_windows


def _angular_n_windows(window_spacing: float) -> float:
    r"""Calculate and return the number of angular windows.

    this is the :math:`N_{\theta }` term in equation 10 of the paper's
    online method, which we've rearranged in order to get this.

    Parameters
    ----------
    window_spacing
        The spacing of the polar angle windows.

    Returns
    -------
    n_windows
        The number of windows that fit into 2 pi.

    """
    return (2 * np.pi) / window_spacing


def _eccentricity_window_spacing(
    min_ecc: float = 0.5,
    max_ecc: float = 15,
    n_windows: float | None = None,
    scaling: float | None = None,
    std_dev: float | None = None,
) -> float:
    r"""Calculate and return the window spacing for the eccentricity windows.

    this is the :math:`w_e` term in equation 11 of the paper's online
    methods (referred to as the window width), which we also refer to as
    the radial spacing. Note that we take exactly one of ``n_windows``
    or ``scaling`` in order to determine this value.

    If scaling is set, ``min_ecc`` and ``max_ecc`` are ignored (the window
    width only depends on scaling, not also on the range of eccentricities;
    they only matter when determining the width using ``n_windows``)

    For both cosine and gaussian windows, this is the distance between
    the peaks of the windows. For cosine windows, this is also the same
    as the windows' widths, but gausian windows' widths are
    approximately ``window_spacing * std_dev * 3`` (since they're
    Gaussian, 99.73% of their mass lie within 3 standard deviations, but
    the Gaussians are technically infinite); but remember that these
    values are in log space not linear.

    Parameters
    ----------
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    n_windows
        The number of log-eccentricity windows we create. ``n_windows``
        xor ``scaling`` must be set.
    scaling
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity (see the ``scaling``
        function). ``n_windows`` xor ``scaling`` must be set.
    std_dev
        The standard deviation of the Gaussian window. If this is set,
        we compute the scaling value for the Gaussian windows instead of
        for the cosine ones.

    Returns
    -------
    window_spacing
        The spacing  of the log-eccentricity windows.

    Raises
    ------
    Exception
        If ``n_windows`` or ``scaling`` is not set

    Notes
    -----
    No equation was given in the paper to calculate the window spacing,
    :math:`w_e` from the scaling, :math:`s`, so we derived it
    ourselves. We start with the final equation for the scaling, given
    in the Notes for the ``scaling`` function.

    .. math::

        s &= \exp(w_e x_h) - \exp(-w_e x_h) \\
        s &= \exp(w_e x_h) - \frac{1}{\exp(w_e x_h)} \\

    We then substitute :math:`t=\exp(w_e x_h)`

    .. math::

        s &= t - \frac{1}{t}
        0 &= t^2 - st - 1

    Then using the quadratic formula:

    .. math::

        t &= \frac{s \pm \sqrt{s^2+4}}{2}

    We then substitute back for :math:`t` and drop the negative root
    because the window spacing is strictly positive.

    .. math::

        \exp(w_e x_h) &= \frac{s + \sqrt{s^2 + 4}}{2}
        w_e &= \log(\frac{s+\sqrt{s^2+4}}{2}) / x_h

    """
    if scaling is not None:
        x_half_max = 0.5 if std_dev is None else std_dev * np.sqrt(2 * np.log(2))
        spacing = np.log((scaling + np.sqrt(scaling**2 + 4)) / 2) / x_half_max
    elif n_windows is not None:
        spacing = (np.log(max_ecc) - np.log(min_ecc)) / n_windows
    else:
        raise Exception("Exactly one of n_windows or scaling must be set!")
    return spacing


def _eccentricity_n_windows(
    window_spacing: float,
    min_ecc: float = 0.5,
    max_ecc: float = 15,
    std_dev: float | None = None,
) -> float:
    r"""Calculate and return the number of eccentricity windows.

    this is the :math:`N_e` term in equation 11 of the paper's online
    method, which we've rearranged in order to get this.

    Parameters
    ----------
    window_spacing
        The spacing of the log-eccentricity windows.
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    std_dev
        The standard deviation of the Gaussian window. Adds extra
        windows to account for the fact that Gaussian windows are
        larger. If using cosine windows, this should be None

    Returns
    -------
    n_windows
        The number of log-eccentricity windows we create.

    """
    n_windows = (np.log(max_ecc) - np.log(min_ecc)) / window_spacing
    # the Gaussians need extra windows in order to make sure that we're
    # summing to 1 across the whole image (because they're wider and
    # shorter). to make sure of this, we want to get all the windows
    # past it up til the one who is 5 standard deviations away from the
    # outermost window calculated above (this matters more for larger
    # values of std_dev / larger windows).
    if std_dev is not None:
        n_windows += 5 * std_dev
    return n_windows


def scaling(
    n_windows: float,
    min_ecc: float = 0.5,
    max_ecc: float = 15,
    std_dev: float | None = None,
) -> float:
    r"""Calculate and return the scaling value, as reported in the paper.

    Scaling is the ratio of the eccentricity window's radial full-width
    at half-maximum to eccentricity. For eccentricity, we use the
    window's "central eccentricity", the one where the input to the
    window function (:math:`x` in equation 9 in the online methods) is 0.

    Parameters
    ----------
    n_windows
        The number of log-eccentricity windows we create.
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    std_dev
        The standard deviation fo the Gaussian window. If this is set,
        we compute the scaling value for the Gaussian windows instead of
        for the cosine ones.

    Returns
    -------
    scaling
        The ratio of the eccentricity window's radial full-width at
        half-maximum to eccentricity

    Notes
    -----
    No equation for the scaling, :math:`s`, was included in the paper,
    so we derived this ourselves. To start, we note that the window
    function equation (equation 9) reaches its half-max (.5) at
    :math:`x=\pm .5`, and that, as above, we treat :math:`x=0` as the
    central eccentricity of the window. Then we must solve for these,
    using the values given within the parentheses in equation 11 as the
    value for :math:`x`, and take their ratios.

    In the following equations, we'll use :math:`x_h` as the value at
    which the window reaches its half-max. For the cosine windows, this
    is always :math:`\pm .5`, but for the Gaussian windows, it's
    :math:`x_h=\sigma\sqrt{2\log 2}`.

    It turns out that this holds for all permissible values of
    ``transition_region_width`` (:math:`t` in the equations) (try
    playing around with some plots if you don't believe me).

    Full-width half-maximum, :math:`W`, the difference between the two
    values of :math:`e_h`:

    .. math::

        \pm x_h &= \frac{\log(e_h) - (log(e_0)+w_e(n+1))}{w_e} \\
        e_h &= e_0 \cdot \exp(w_e(\pm x_h+n+1)) \\
        W &= e_0 (\exp(w_e(n+1+x_h)) - \exp(w_e(n+1-x_h))

    Window's central eccentricity, :math:`e_c`:

    .. math::

        0 &= \frac{\log(e_c) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_c &= e_0 \cdot \exp(w_e(n+1))

    Then the scaling, :math:`s` is the ratio :math:`\frac{W}{e_c}`:

    .. math::

        s &= \frac{e_0 (\exp(w_e(n+1+x_h)) -
        exp(w_e(n+1-x_h)))}{e_0 \cdot \exp(w_e(n+1))} \\
        s &= \frac{\exp(w_e(n+1+x_h))}{\exp(w_e(n+1))} -
        \frac{\exp(w_e(n+1-x_h))}{\exp(w_e(n+1))} \\
        s &= \exp(w_e(n+1+x_h-n-1)) - \exp(w_e(n+1-x_h-n-1)) \\
        s &= \exp(x_h\cdot w_e) - \exp(-x_h\cdot w_e)

    Note that we don't actually use the value returned by
    ``_windows_eccentricity`` for :math:`e_c`; we simplify
    it away in the calculation above.

    """
    x_half_max = std_dev * np.sqrt(2 * np.log(2)) if std_dev is not None else 0.5
    window_spacing = _eccentricity_window_spacing(min_ecc, max_ecc, n_windows)
    return np.exp(x_half_max * window_spacing) - np.exp(-x_half_max * window_spacing)


def _windows_eccentricity(
    ecc_type: Literal["min", "central", "max", "{n}std"],
    n_windows: float,
    window_spacing: float,
    min_ecc: float = 0.5,
    transition_region_width: float = 0.5,
    std_dev: float | None = None,
) -> np.ndarray:
    r"""Calculate a relevant eccentricity for each radial window.

    These are the values :math:`e_c`, as referred to in ``scaling``
    (for each of the n windows)

    Parameters
    ----------
    ecc_type
        Which eccentricity you want to calculate: the minimum one where
        x=-(1+t)/2, the central one where x=0, or the maximum one where
        x=(1+t)/2. if std_dev is set, minimum and maximum are +/- 3
        std_dev. if '{n}std' (where n is a positive or negative
        integer), then we return the eccentricity at that many std_dev
        away from center (only std_dev is set).
    n_windows
        The number of log-eccentricity windows we create. n_windows can
        be a non-integer, in which case we round it up (thus one of our
        central eccentricities might be above the maximum eccentricity
        for the windows actually created)
    window_spacing
        The spacing of the log-eccentricity windows.
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    transition_region_width
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. Must lie between 0 and 1.
    std_dev
        The standard deviation fo the Gaussian window. If this is set,
        we compute the eccentricities for the Gaussian windows instead of
        for the cosine ones.

    Returns
    -------
    eccentricity
        A list of length ``n_windows``, containing the minimum, central,
        or maximum eccentricities of each window.

    Raises
    ------
    Exception
        If ``ecc_type`` takes an illegal value.

    Notes
    -----
    For the raised-cosine windows, to find 'min', we solve for the
    eccentricity where :math:`x=\frac{-(1+t)}{2}` in equation 9:

    .. math::

        \frac{-(1+t)}{2} &= \frac{\log(e_{min}) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_{min} &= \exp{\frac{-w_e(1+t)}{2} + \log{e_0} + w_e(n+1)}

    To find 'max', we solve for the eccentricity where
    :math:`x=\frac{(1+t)}{2}` in equation 9:

    .. math::

        \frac{(1+t)}{2} &= \frac{\log(e_{max}) -(\log(e_0)+w_e(n+1))}{w_e} \\
        e_{max} &= \exp{\frac{w_e(1+t)}{2} + \log(e_0) + w_e(n+1)}

    For either raised-cosine or gaussian windows, to find 'central', we
    solve for the eccentricity where :math:`x=0` in equation 9:

    .. math::

        0 &= \frac{\log(e_c) -(log(e_0)+w_e(n+1))}{w_e} \\
        e_c &= e_0 \cdot \exp(w_e(n+1))

    For the gaussian windows, we say min and max are at :math:`x=\pm 3
    \sigma`, respectively:

    .. math::

        3 \sigma &= \frac{\log(e_{max}) - (\log(e_0) + w_e(n+1))}{w_e}
        e_{max} &= \exp{3\sigma w_e + \log(e_0) + w_e(n+1)}

    And, similarly:

    .. math::

        -3 \sigma &= \frac{\log(e_{min}) - (\log(e_0) + w_e(n+1))}{w_e}
        e_{min} &= \exp{-3\sigma w_e + \log(e_0) + w_e(n+1)}

    """
    if ecc_type not in ["min", "max", "central"] and not ecc_type.endswith("std"):
        raise Exception(f"Don't know how to handle ecc_type {ecc_type}")
    if ecc_type == "central":
        ecc = [
            min_ecc * np.exp(window_spacing * (i + 1))
            for i in np.arange(np.ceil(n_windows))
        ]
    elif ecc_type == "min":
        if std_dev is None:
            ecc = [
                (
                    np.exp(-window_spacing * (1 + transition_region_width) / 2)
                    * min_ecc
                    * np.exp(window_spacing * (i + 1))
                )
                for i in np.arange(np.ceil(n_windows))
            ]
        else:
            ecc = [
                (
                    np.exp(-3 * std_dev * window_spacing)
                    * min_ecc
                    * np.exp(window_spacing * (i + 1))
                )
                for i in np.arange(np.ceil(n_windows))
            ]
    elif ecc_type == "max":
        if std_dev is None:
            ecc = [
                (
                    np.exp(window_spacing * (1 + transition_region_width) / 2)
                    * min_ecc
                    * np.exp(window_spacing * (i + 1))
                )
                for i in np.arange(np.ceil(n_windows))
            ]
        else:
            ecc = [
                (
                    np.exp(3 * std_dev * window_spacing)
                    * min_ecc
                    * np.exp(window_spacing * (i + 1))
                )
                for i in np.arange(np.ceil(n_windows))
            ]
    elif ecc_type.endswith("std"):
        if std_dev is None:
            raise Exception(f"std_dev must be set if ecc_type == {ecc_type}")
        else:
            n = int(re.findall("([-0-9]+)std", ecc_type)[0])
            ecc = [
                (
                    np.exp(n * std_dev * window_spacing)
                    * min_ecc
                    * np.exp(window_spacing * (i + 1))
                )
                for i in np.arange(np.ceil(n_windows))
            ]
    return np.array(ecc)


def _window_widths_actual(
    angular_window_spacing: float,
    radial_window_spacing: float,
    min_ecc: float = 0.5,
    max_ecc: float = 15,
    window_type: Literal["cosine", "gaussian"] = "gaussian",
    transition_region_width: float | None = None,
    std_dev: float | None = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate and return the actual widths of the windows.

    whereas ``_angular_window_spacing`` returns a term used in the
    equations to generate the windows, this returns the actual angular
    and radial widths of each set of windows (in degrees).

    We return four total widths, two by two for radial and angular by
    'top' and 'full'. By 'top', we mean the width of the flat-top region
    of each window (where the windows value is 1), and by 'full', we
    mean the width of the entire window

    Parameters
    ----------
    angular_window_spacing
        The width of the windows in the angular direction, as returned
        by ``_angular_window_spacing``
    radial_window_spacing
        The width of the windows in the radial direction, as returned by
        ``_eccentricity_window_spacing``
    min_ecc
        The minimum eccentricity, the eccentricity below which we do not
        compute pooling windows (in degrees). Parameter :math:`e_0` in
        equation 11 of the online methods.
    max_ecc
        The maximum eccentricity, the outer radius of the image (in
        degrees). Parameter :math:`e_r` in equation 11 of the online
        methods.
    window_type
        Whether to use the raised cosine function from [1]_ or a
        Gaussian that has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then
        ``std_dev`` must be set
    transition_region_width
        The width of the cosine windows' transition region, parameter
        :math:`t` in equation 9 from the online methods.
    std_dev
        The standard deviation of the Gaussian window.

    Returns
    -------
    radial_top_width
        The width of the flat-top region of the windows in the radial
        direction (each value corresponds to a different ring of
        windows, from the fovea to the periphery).
    radial_full_width
        The full width of the windows in the radial direction (each
        value corresponds to a different ring of windows, from the fovea
        to the periphery).
    angular_top_width
        The width of the flat-top region of the windows in the angular
        direction (each value corresponds to a different ring of
        windows, from the fovea to the periphery).
    angular_full_width
        The full width of the windows in the angular direction (each
        value corresponds to a different ring of windows, from the fovea
        to the periphery).

    Notes
    -----
    For raised-cosine windows:

    In order to calculate the width in the angular direction, we start
    with the angular window width (:math:`w_{\theta }`). The 'top' width
    is then :math:`w_{\theta}(1-t)` and the 'full' width is
    :math:`w_{\theta}(1+t)`, where :math:`t` is the
    ``transition_region_width``. This gives us the width in radians, so
    we convert it to degrees by finding the windows' central
    eccentricity (:math:`e_c`, as referred to in ``scaling`` and
    returned by ``_windows_eccentricity``), and find the
    circumference (in degrees) of the circle that goes through that
    eccentricity. We then multiply our width in radians by
    :math:`\frac{2\pi e_c}{2\pi}=e_c`.

    Calculating the width in the radial direction is slightly more
    complicated, because they're not symmetric or constant across the
    visual field. We start by noting, based on equation 9 in the paper,
    that the flat-top region is the region between :math:`x=\frac{\pm
    (1-t)}{2}` and the whole window is the region between
    :math:`x=\frac{\pm (1+t)}{2}`. We can then do a little bit of
    rearranging to forms used in this function.

    For gaussian windows:

    The 'top' width in either direction is 0, because a gaussian has no
    flat top region.

    We consider the 'full' width to be 3 standard deviations out. That
    means that a given window's full extent goes from :math:`x=-3\sigma`
    to :math:`x=3\sigma`, where :math:`\sigma` is ``std_dev``, the
    window's standard deviation, and :math:`x` is the input to the
    ``gaussian`` function (analogous to the ``raised_cosine`` function).

    In the angular direction, for window :math:`n`,
    :math:`x=\frac{\theta-w_\theta n}{w_\theta}` (see equation 10, and
    we ignore the part of that equation that includes :math:`t`, because
    the gaussian windows have no transition region width). Rearranging,
    we see that the extent of the window in radians is thus :math:`\pm
    3\sigma w_\theta`, so the full width is then :math:`6\sigma w_\theta
    e_c`, where :math:`e_c` is the window's central eccentricity and
    necessary to convert it to degreess.

    We can follow similar logic for the radial direction, knowing that
    we want to find the difference between math:`\exp(\pm 3\sigma w_e +
    \log e_0 + w_e(n+1))` and rearranging to the forms used in this
    function.

    """
    n_radial_windows = np.ceil(
        _eccentricity_n_windows(radial_window_spacing, min_ecc, max_ecc, std_dev)
    )
    window_central_eccentricities = _windows_eccentricity(
        "central", n_radial_windows, radial_window_spacing, min_ecc
    )
    if window_type == "cosine":
        radial_top = [
            min_ecc
            * (
                np.exp(
                    (radial_window_spacing * (3 + 2 * n - transition_region_width)) / 2
                )
                - np.exp(
                    (radial_window_spacing * (1 + 2 * n + transition_region_width)) / 2
                )
            )
            for n in np.arange(n_radial_windows)
        ]
        radial_full = [
            min_ecc
            * (
                np.exp(
                    (radial_window_spacing * (3 + 2 * n + transition_region_width)) / 2
                )
                - np.exp(
                    (radial_window_spacing * (1 + 2 * n - transition_region_width)) / 2
                )
            )
            for n in np.arange(n_radial_windows)
        ]
        angular_top = [
            angular_window_spacing * (1 - transition_region_width) * e_c
            for e_c in window_central_eccentricities
        ]
        angular_full = [
            angular_window_spacing * (1 + transition_region_width) * e_c
            for e_c in window_central_eccentricities
        ]
    elif window_type == "gaussian":
        # gaussian windows have no flat top region, so this is always 0
        radial_top = [0 for i in np.arange(n_radial_windows)]
        angular_top = [0 for i in np.arange(n_radial_windows)]
        radial_full = [
            min_ecc
            * (
                np.exp(radial_window_spacing * (3 * std_dev + n + 1))
                - np.exp(radial_window_spacing * (-3 * std_dev + n + 1))
            )
            for n in np.arange(n_radial_windows)
        ]
        angular_full = [
            6 * std_dev * angular_window_spacing * e_c
            for e_c in window_central_eccentricities
        ]
    return (
        np.array(radial_top),
        np.array(radial_full),
        np.array(angular_top),
        np.array(angular_full),
    )


def deg_to_pix(img_res: tuple[int, int], max_eccentricity: float = 15) -> float:
    r"""Calculate the degree-to-pixel conversion factor.

    We assume ``img_res`` is the full resolution of the image and
    ``max_eccentricity`` is the radius of the image in degrees. Thus, we
    divide half of ``img_res`` by ``max_eccentricity``. However, we want
    to be able to handle non-square images, so we assume the value you
    want to use is the max of the two numbers in ``img_res`` (this is
    the way we construct the PoolingWindow objects; we want the windows
    to fill the full image).

    Parameters
    ----------
    img_res
        The resolution of our image (should therefore contains
        integers).
    max_eccentricity
        The eccentricity (in degrees) of the edge of the image

    Returns
    -------
    deg_to_pix
        The factor to convert degrees to pixels (in
        pixels/degree). E.g., multiply the eccentricity (in degrees) by
        deg_to_pix to get it in pixels

    """
    return (np.max(img_res) / 2) / max_eccentricity


def _min_eccentricity(
    scaling: float,
    img_res: tuple[int, int],
    max_eccentricity: float = 15,
    pixel_area_thresh: float = 1,
    radial_to_circumferential_ratio: float = 2,
) -> tuple[float, float]:
    r"""Calculate the eccentricity where window area exceeds a threshold.

    The pooling windows are used primarily for metamer synthesis, and
    conceptually, if the pooling windows only include a single pixel, or
    smaller, then the only metamer for that window is exactly that
    pixel. Therefore, we don't need to compute these tiny windows, and
    we don't want to for computational reasons (it will make everything
    take much longer).

    Since ``scaling`` sets the size of the windows at eccentricity, by
    giving the slope between the diameter (in the radial direction) at
    half-max and the eccentricity, we can use it to determine the area
    of the window in each direction. What we calculate here is only an
    approximation and slightly arbitrary. Let :math:`s` be the scaling
    value, then we approximate the area :math:`(s \cdot e \cdot
    \frac{r_{pix}}{r_{deg}})^2 \cdot \frac{\pi}{4} \cdot \frac{1}{r}`,
    where :math:`r` is the ratio between the radial and circumferential
    widths, i.e. the ``radial_to_circumferential_ratio``
    arg. :math:`s\cdot e` is the diameter at half-max in degrees,
    multiplying it by :math:`\frac{r_{pix}}{r_{deg}}` (the radius of the
    image in pixels and degrees, respectively) converts it to pixels,
    and that and multiplying by :math:`\frac{pi}{4}` gives the area of a
    circle with that diameter; multiplying it by :math:`\frac{1}{r}`
    then converts this to a regular oval with this aspect ratio. This is
    a lower-bound on the area of our windows, which are actually
    elongated ovals with a larger radius than this, and thus a bit more
    complicated to compute.

    Note that, since we're using the scaling to figure this out, we're
    computing the area at approximately the windows' full-max
    half-width, and this is what we're doing for both gaussian and
    raised-cosine windows (though gaussian windows technically extend
    further beyond the FWHM than raised-cosine windows, the difference
    is not large for small windows).

    Parameters
    ----------
    scaling
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
    img_res
        The resolution of our image (should therefore contains
        integers).
    max_eccentricity
        The eccentricity (in degrees) of the edge of the image
    pixel_area_thresh
        What area (in square pixels) to check against our approximate
        pooling window area. This is slightly arbitrary, but should be
        consistent
    radial_to_circumferential_ratio
        ``scaling`` determines the number of log-eccentricity windows we
        can create; this ratio gives us the number of polar angle
        ones. Based on `scaling`, we calculate the width of the windows
        in log-eccentricity, and then divide that by this number to get
        their width in polar angle. Because we require an integer number
        of polar angle windows, we round the resulting number of polar
        angle windows to the nearest integer, so the ratio in the
        generated windows approximate this. 2 (the default) is the value
        used in the paper [1]_.

    Returns
    -------
    min_ecc_deg
        The eccentricity (in degrees) where window area will definitely
        exceed ``pixel_area_thresh``
    min_ecc_pix
        The eccentricity (in pixels) where window area will definitely
        exceed ``pixel_area_thresh``

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    degtopix = deg_to_pix(img_res, max_eccentricity)
    # see docstring for why we use this formula, but we're computing the
    # coefficients of a quadratic equation as a function of eccentricity
    # and use np.roots to find its roots
    quad_coeff = (
        (scaling * degtopix) ** 2 * (np.pi / 4) / radial_to_circumferential_ratio
    )
    # we only want the positive root
    min_ecc_deg = np.max(np.roots([quad_coeff, 0, -pixel_area_thresh]))
    return min_ecc_deg, min_ecc_deg * degtopix
