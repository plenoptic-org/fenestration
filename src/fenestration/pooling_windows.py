"""Contains the PoolingWindows class.

This is the module you should use to get the pooling used in models like those
found in Freeman and Simoncelli, 2011

pooling.py and calculate.py contain a lot of necessary functions

"""

import itertools
import os.path as op
import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
import torch
from matplotlib.figure import Figure
from torch import nn

from . import _plot, _tensors, calculate, create_pooling_windows, pooling

__all__ = [
    "PoolingWindows",
]


def __dir__() -> list[str]:
    return __all__


class PoolingWindows(nn.Module):
    r"""Generic class to set up and visualize foveated pooling windows.

    This generates foveated pooling windows given a small number of
    parameters. These windows are organized radially into eccentricity
    bands with the size, shape, and extent of the windows dependent upon
    the input parameters. These pooling windows can be used to summarize
    model statistics across visual space, such that information near the
    central (or foveal) visual field is pooled over smaller regions whereas
    information near the outer (or peripheral) visual field is pooled over
    larger regions.

    One tricky thing we do is generate a set of scaling
    windows for each (appropriately-sized) scale. For example, a V1
    model may have 4 scales, so for a 256 x 256 image, the coefficients
    will have shape (256, 256), (128, 128), (64, 64), and (32,
    32). Therefore, we need windows of the same size (could also
    up-sample the coefficient tensors, but since that would need to
    happen each iteration of the metamer synthesis, pre-generating
    appropriately sized windows is more efficient).

    We will calculate the minimum eccentricity at which the area of the
    windows at half-max exceeds one pixel at each scale. For scales
    beyond the first however, we will not throw an Exception if this
    value is below :attr:`min_ecc`. We instead print a warning to
    alert the user and use this value as :attr:`min_ecc` when
    creating the plots. In order to see what this value was, see
    :attr:`calculated_min_eccentricity_pixels`.

    We can optionally cache the windows tensor we create, if
    :attr:`cache_dir` is not ``None``. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_ecc}_
    em-{max_ecc}_{window_type}.pt``. We'll cache each scale separately,
    changing the :attr:`img_res` (and potentially :attr:`min_ecc`) values in that save
    path appropriately.

    Parameters
    ----------
    scaling
        Scaling parameter that governs the size of the pooling
        windows.
    img_res
        The resolution of our image (should therefore contains
        integers). Will use this to generate appropriately sized pooling
        windows where :attr:`max_ecc` is set to the outer radius of the image.
    min_ecc
        The eccentricity at which the pooling windows start.
    max_ecc
        The eccentricity at which the pooling windows end.
    num_scales
        The number of scales to generate masks for. For the RGC model,
        this should be 1, otherwise should match the number of scales in
        the steerable pyramid.
    cache_dir
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    window_type
        Whether to use the raised cosine function from [1]_ or a Gaussian that
        has approximately the same structure.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    img_res : tuple
        The resolution of our image in pixels.
    min_ecc : float
        The eccentricity at which the pooling windows start.
    max_ecc : float
        The eccentricity at which the pooling windows end.
    angle_windows : dict
        A dict of 3d tensors containing the angular pooling windows in
        which the model parameters are averaged. Each key corresponds to
        a different scale and thus is a different size.
    ecc_windows : dict
        A dict of 3d tensors containing the log-eccentricity pooling
        windows in which the model parameters are averaged. Each key
        in the dict corresponds to a different scale and thus is a
        different size.
    norm_factor : dict
        A dict of 3d tensors containing the values used to normalize
        :attr:`ecc_windows`. Each key corresponds to a different scale. This is
        stored to undo that normalization for plotting and projection.
    window_width_pixels : list
        List of dictionaries containing the widths of the windows in
        pixels; each entry in the list corresponds to the widths for a
        different scale, as in ``windows``. See above for explanation of
        the dictionaries. To visualize these, see the
        :meth:`plot_window_widths` method.
    n_polar_windows : int
        The number of windows we have in the polar angle dimension
        (within each eccentricity band)
    n_eccentricity_bands : int
        The number of eccentricity bands in our model
    calculated_min_eccentricity_pixels : list
        List of floats (one for each scale) that contain
        the minimum eccentricity (in pixels) where the area of the window
        at half-max exceeds one pixel (based on the scaling, size of the
        image in pixels and in degrees).
    central_eccentricity_pixels : list
        List of 1d arrays (one for each scale), each with shape
        ``(self.n_eccentricity_bands,)``, each value gives the
        eccentricity of the center of each eccentricity band of windows.
    window_approx_area_pixels : list
        List of dictionaries containing the approximate areas of the
        windows in pixels; each entry in the list corresponds to the
        areas for a different scale, as in ``windows``. There are three
        keys: 'top', 'half', and 'full', corresponding to which width we
        used to calculate window areas. For cosine windows, top is the width
        of the flat-top region of each window, where the window's value is
        1; full is the width of the entire window; half is the width at
        half-max. For gaussian windows, there is no flat-top region, full is
        3 standard deviations, and half is the width at half max. To get this
        approximate area, we multiply the radial and angular widths against each
        other and then by pi/4 to get the area of the regular ellipse that has
        those widths (our windows are elongated, so this is probably an
        under-estimate). To visualize these, see the :meth:`plot_window_areas` method.
    deg_to_pix : list
        List of floats containing the degree-to-pixel conversion factor
        at each scale
    cache_dir : str or None
        If str, this is the directory where we cached / looked for
        cached windows tensors. This directory must already exist, or we raise
        a FileNotFoundError.
    cache_paths : list
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    num_scales : int
        Number of scales this object has windows for
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a
        Gaussian that has approximately the same structure.
    window_max_amplitude : float
        The max amplitude of an individual window. This will always be 1
        for raised-cosine windows. For gaussian windows, this value depends
        on the standard deviation, which is currently hard-coded at 1.
        Therefore, for gaussian windows it's approximately 0.16.
    window_intersecting_amplitude : float
        The amplitude at which two neighboring windows intersect. This
        will always be .5 for raised-cosine windows, but for gaussian ones,
        this value depends on the standard deviation. This value is currently
        hard-coded at 1, therefore it's half a standard deviation away from
        the center, approximately 0.14.

    Raises
    ------
    ValueError
        If :attr:`img_res` is not 2d

    See Also
    --------
    create_pooling_windows : Create angle and eccentricity window tensors.

    Notes
    -----
    We will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    :attr:`scaling`, :attr:`img_res` and :attr:`max_ecc`) and, if
    :attr:`min_ecc` is below that, will print a warning.

    If you are just interested in the eccentricity and angular filters
    associated with these pooling windows, this is also possible using
    a combination of :mod:`~fenestration.create_pooling_windows` and
    :mod:`~fenestration.pooling.normalize_windows`. See Examples section of
    :mod:`~fenestration.create_pooling_windows` for details on this process.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the
       ventral stream. Nature Neuroscience, 14(9),
       1195–1201. http://dx.doi.org/10.1038/nn.2889

    """

    def __init__(
        self,
        scaling: float,
        img_res: tuple[int, int],
        min_ecc: float = 0.5,
        max_ecc: float = 15,
        num_scales: int = 1,
        cache_dir: str | None = None,
        window_type: Literal["cosine", "gaussian"] = "gaussian",
    ):
        super().__init__()
        if len(img_res) != 2:
            raise ValueError("img_res must be 2d!")
        self.scaling = scaling
        self.min_ecc = float(min_ecc)
        self.max_ecc = float(max_ecc)
        self.img_res = img_res
        self.num_scales = num_scales
        self.window_type = window_type
        self.angle_windows = {}
        self.ecc_windows = {}
        self._contract_expr = {}
        self.norm_factor = {}
        if window_type == "cosine":
            self._transition_region_width = 0.5
            self._std_dev = None
            self.window_max_amplitude = 1
            self.window_intersecting_amplitude = 0.5
        elif window_type == "gaussian":
            self._std_dev = 1
            self._transition_region_width = None
            # 1 / (std_dev * GAUSSIAN_SUM) is the max in a single
            # direction (radial or angular), so the max for a single
            # window is its square
            self.window_max_amplitude = (
                1 / (self._std_dev * pooling.GAUSSIAN_SUM)
            ) ** 2
            self.window_intersecting_amplitude = self.window_max_amplitude * np.exp(
                -0.25 / 2
            )
        else:
            raise ValueError("Window types must be either gaussian or cosine!")
        if cache_dir is not None:
            self.cache_dir = op.expanduser(cache_dir)
            if not op.exists(self.cache_dir):
                raise FileNotFoundError(
                    f"{self.cache_dir} was specified as cache_dir, "
                    "but that directory does not exist!"
                )
            cache_path_template = op.join(
                self.cache_dir,
                "scaling-{scaling}_size-{img_res}_"
                "e0-{min_ecc:.03f}_em-{max_ecc:.01f}_"
                "{window_type}.pt",
            )
        else:
            self.cache_dir = cache_dir
        self.cache_paths = []
        self._calculated_min_eccentricity_degrees = []
        self.calculated_min_eccentricity_pixels = []
        self._window_sizes()
        for i in range(self.num_scales):
            scaled_img_res = [np.ceil(j / 2**i) for j in img_res]
            minimal_ecc, minimal_ecc_pix = calculate._min_eccentricity(
                scaling, scaled_img_res, max_ecc
            )
            self._calculated_min_eccentricity_degrees.append(minimal_ecc)
            self.calculated_min_eccentricity_pixels.append(minimal_ecc_pix)
            if self.min_ecc is not None and minimal_ecc > self.min_ecc:
                warnings.warn(
                    f"Creating windows for scale {i} with min_ecc "
                    f"{self.min_ecc}, but calculated minimal eccentricity is "
                    f"{minimal_ecc}, so be aware some are smaller than a pixel!"
                )
            angle_windows = None
            ecc_windows = None
            if cache_dir is not None:
                format_kwargs = dict(
                    scaling=scaling,
                    max_ecc=self.max_ecc,
                    img_res=",".join([str(int(i)) for i in scaled_img_res]),
                    window_type=window_type,
                    min_ecc=self.min_ecc,
                )
                self.cache_paths.append(cache_path_template.format(**format_kwargs))
                if op.exists(self.cache_paths[-1]):
                    warnings.warn(f"Loading windows from cache: {self.cache_paths[-1]}")
                    windows = torch.load(self.cache_paths[-1])
                    angle_windows = windows["angle"]
                    ecc_windows = windows["ecc"]
            if angle_windows is None or ecc_windows is None:
                angle_windows, ecc_windows = create_pooling_windows(
                    scaling,
                    scaled_img_res,
                    self.min_ecc,
                    self.max_ecc,
                    std_dev=self._std_dev,
                    transition_region_width=self._transition_region_width,
                    window_type=window_type,
                )

                if cache_dir is not None:
                    warnings.warn(f"Saving windows to cache: {self.cache_paths[-1]}")
                    torch.save(
                        {"angle": angle_windows, "ecc": ecc_windows},
                        self.cache_paths[-1],
                    )
            # the observer model requires every scale has the same number of
            # eccentricity windows, so we add empty windows to make sure that's
            # the case. this is only an issue with coarse scales (or
            # equivalently, small resolutions) and smaller scales
            if ecc_windows.shape[0] < self._central_eccentricity_degrees.shape[0]:
                n_extra_wdws = (
                    self._central_eccentricity_degrees.shape[0] - ecc_windows.shape[0]
                )
                ecc_windows = torch.cat(
                    [ecc_windows, torch.zeros_like(ecc_windows[:n_extra_wdws])]
                )
            self.angle_windows[i] = angle_windows
            self.ecc_windows[i] = ecc_windows
            # cache this calculation so the forward() call will be a bit
            # faster. we don't know how many elements we'll have in the batch
            # or channel dimension (so these are just dummy numbers), but there
            # would need to be a *huge* amount of them to change the most
            # efficient way of doing this.
            self._contract_expr[i] = oe.contract_expression(
                "bchw,ahw,ehw->bcea",
                (1, 5, *angle_windows.shape[1:]),
                angle_windows.shape,
                ecc_windows.shape,
            )
            # if we have the eccentricity one std dev away from center, use
            # that.
            try:
                ecc = self.one_std_dev_eccentricity_degrees
            # otherwise, use the central one.
            except AttributeError:
                ecc = self._central_eccentricity_degrees
            norm_ecc, norm_factor = pooling.normalize_windows(
                angle_windows, ecc_windows, ecc
            )
            self.ecc_windows[i] = norm_ecc
            self.norm_factor[i] = norm_factor

    def _window_sizes(self):
        r"""Calculate the various window size metrics.

        Helper function that gets called during construction, should not
        be used by user. Sets the following attribute: n_polar_windows,
        n_eccentricity_bands, _window_width_degrees, _central_eccentricity_degrees,
        _window_approx_area_degrees, window_width_pixels, central_eccentricity_pixels,
        window_approx_area_pixels, deg_to_pix

        All of these are based on calling various helper functions (from
        ``fen.calculate``) and doing simple calculations
        based on the attributes already set (largely min_ecc,
        max_ecc, scaling, and transition_region_width)

        """
        ecc_window_width = calculate._eccentricity_window_spacing(
            scaling=self.scaling, std_dev=self._std_dev
        )
        n_polar_windows = int(round(calculate._angular_n_windows(ecc_window_width / 2)))
        self.n_polar_windows = n_polar_windows
        angular_window_width = calculate._angular_window_spacing(self.n_polar_windows)
        # we multiply max_ecc by sqrt(2) here because we want
        # to go out to the corner of the image
        window_widths = calculate._window_widths_actual(
            angular_window_width,
            ecc_window_width,
            self.min_ecc,
            self.max_ecc * np.sqrt(2),
            self.window_type,
            self._transition_region_width,
            self._std_dev,
        )
        self._window_width_degrees = dict(
            zip(
                ["radial_top", "radial_full", "angular_top", "angular_full"],
                window_widths,
            )
        )
        self.n_eccentricity_bands = len(self._window_width_degrees["radial_top"])
        # transition width and std dev don't matter for central
        # eccentricity, just min and max
        self._central_eccentricity_degrees = calculate._windows_eccentricity(
            "central",
            self.n_eccentricity_bands,
            ecc_window_width,
            self.min_ecc,
        )
        if self.window_type == "gaussian":
            self.one_std_dev_eccentricity_degrees = calculate._windows_eccentricity(
                "1std",
                self.n_eccentricity_bands,
                ecc_window_width,
                self.min_ecc,
                std_dev=self._std_dev,
            )
        self._window_width_degrees["radial_half"] = (
            self.scaling * self._central_eccentricity_degrees
        )
        # the 2 we divide by here is the
        # radial_to_circumferential_ratio; if we ever allow that to be
        # set by the user will need to update
        self._window_width_degrees["angular_half"] = (
            self._window_width_degrees["radial_half"] / 2
        )
        self._window_approx_area_degrees = {}
        for k in ["full", "top", "half"]:
            self._window_approx_area_degrees[k] = (
                self._window_width_degrees[f"radial_{k}"]
                * self._window_width_degrees[f"angular_{k}"]
                * (np.pi / 4)
            )
        self.window_width_pixels = []
        self.window_approx_area_pixels = []
        self.central_eccentricity_pixels = []
        self.deg_to_pix = []
        for i in range(self.num_scales):
            deg_to_pix = calculate.deg_to_pix(
                [j / 2**i for j in self.img_res], self.max_ecc
            )
            self.deg_to_pix.append(deg_to_pix)
            self.window_width_pixels.append(
                dict(
                    (k, v * deg_to_pix)
                    for k, v in self._window_width_degrees.copy().items()
                )
            )
            self.window_approx_area_pixels.append({})
            for k in ["full", "top", "half"]:
                self.window_approx_area_pixels[-1][k] = (
                    self.window_width_pixels[-1][f"radial_{k}"]
                    * self.window_width_pixels[-1][f"angular_{k}"]
                    * (np.pi / 4)
                )
            self.central_eccentricity_pixels.append(
                self.deg_to_pix[-1] * self._central_eccentricity_degrees
            )

    def to(self, *args: Any, **kwargs: Any) -> nn.Module:
        r"""Move and/or cast the parameters and buffer.

        This can be called as

        .. code:: python

            to(device=None, dtype=None, non_blocking=False)

        .. code:: python

            to(dtype, non_blocking=False)

        .. code:: python

            to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired ``dtype`` s. In addition, this method will
        only cast the floating point parameters and buffers to ``dtype``
        (if given). The integral parameters and buffers will be moved to
        ``device``, if that is given, but with ``dtype`` s unchanged. When
        ``non_blocking`` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Parameters
        ----------
        device : :class:`torch.device`
            The desired device of the parameters and buffers in this module
        dtype : :class:`torch.dtype`
            The desired floating point type of the floating point parameters
            and buffers in this module
        tensor : :class:`torch.Tensor`
            Tensor whose dtype and device are the desired dtype and device
            for all parameters and buffers in this module

        Returns
        -------
            Module
        """
        for k, v in self.angle_windows.items():
            self.angle_windows[k] = v.to(*args, **kwargs)
        for k, v in self.ecc_windows.items():
            self.ecc_windows[k] = v.to(*args, **kwargs)
        for k, v in self.norm_factor.items():
            self.norm_factor[k] = v.to(*args, **kwargs)
        return self

    def merge(self, other_PoolingWindows: nn.Module, scale_offset: float = 0.5):
        r"""Merge with a second PoolingWindows object.

        This combines the angle_windows, ecc_windows, and window_size
        dictionaries of two PoolingWindows objects. Since they will both
        have similarly-indexed keys (0, 1, 2,... based on
        :attr:`num_scales`), we need some offset to keep them separate,
        which ``scale_offset`` provides. We thus merge the dictionaries like
        so:

        .. code-block:: python

            for k, v in other_PoolingWindows.angle_windows.items():
                self.angle_windows[k + scale_offset] = v

        and similarly for :attr:`ecc_windows` and :attr:`norm_factor`.

        The intended use case for this is to create one PoolingWindows
        object for a steerable pyramid with some number of scales, and
        then a second one for a corresponding "half-octave" steerable
        pyramid, which is built on the original image down-sampled by a
        factor of :math:`\sqrt{2}` in order to sample the frequencies half-way
        between the scales of the original pyramid. You might want to
        slightly adjust the shape of the down-sampled image (e.g., to
        make its size even), so we don't provide support to
        automatically create the windows for the half-scales; instead
        you should create a new PoolingWindows object based on your
        intended size and merge it into the original.

        .. note::
            This method modifies the module in-place.

        Parameters
        ----------
        other_PoolingWindows : fenestration.PoolingWindows
            A second instantiated PoolingWindows object
        scale_offset : float, optional
            The amount to offset all the keys of the second
            PoolingWindows object by (see above for greater explanation)

        """
        for k, v in other_PoolingWindows.angle_windows.items():
            self.angle_windows[k + scale_offset] = v
        for k, v in other_PoolingWindows.ecc_windows.items():
            self.ecc_windows[k + scale_offset] = v
        for k, v in other_PoolingWindows.norm_factor.items():
            self.norm_factor[k + scale_offset] = v

    def forward(
        self, x: dict | torch.Tensor, idx: int = 0, weights: torch.Tensor | None = None
    ) -> dict[torch.Tensor] | torch.Tensor:
        r"""Window and pool the input.

        We take an input, either a 4d tensor or a dictionary of 4d
        tensors, and return the pooled window averages. If it's a 4d
        tensor, we return a 3d tensor, with windows indexed along the
        3rd dimension. If it's a dictionary, we return a dictionary with
        the same keys and have changed all the values to 3d tensors,
        with windows indexed along the 3rd dimension.

        If it's a 4d tensor, we use the ``idx`` entry in the ``windows``
        list. If it's a dictionary, we assume it's keys are ``(scale,
        orientation)`` tuples and so use ``windows[key[0]]`` to find the
        appropriately-sized window (this is the case for, e.g., the
        steerable pyramid). If we want to use differently-structured
        dictionaries, we'll need to restructure this.

        This is equivalent to calling ``self.pool(self.window(x, idx),
        idx)``, however, we don't produce the intermediate products and
        so this is more efficient.

        Parameters
        ----------
        x
            Either a 4d tensor or a dictionary of 4d tensors.
        idx
            Which entry in the ``windows`` list to use. Only used if
            ``x`` is a tensor
        weights
            If not None, should be a tensor of shape (scales, batch, channel,
            eccentricity, angle), this allows us to reweight the pooled input
            across scales, eccentricity, and angle. If None, don't reweight.

        Returns
        -------
        pooled_x
            Same type as ``x``, see above for how it's created.

        See Also
        --------
        window : window the input
        pool : pool the windowed input (get the weighted average)
        project : the opposite of this, going from pooled values to
            image

        """
        if weights is None:
            weights = torch.ones(
                self.num_scales,
                device=self.angle_windows[0].device,
                dtype=self.angle_windows[0].dtype,
            )
        if isinstance(x, dict):
            pooled_x = dict(
                (
                    k,
                    (
                        weights[k[0]]
                        * self._contract_expr[k[0]](
                            v,
                            self.angle_windows[k[0]],
                            self.ecc_windows[k[0]],
                            backend="torch",
                        )
                    ).flatten(2, 3),
                )
                for k, v in x.items()
            )
        else:
            pooled_x = self._contract_expr[idx](
                x, self.angle_windows[idx], self.ecc_windows[idx], backend="torch"
            )
            pooled_x = (weights[idx] * pooled_x).flatten(2, 3)
        return pooled_x

    def window(
        self, x: dict[torch.Tensor] | torch.Tensor, idx: int = 0
    ) -> dict[torch.Tensor] | torch.Tensor:
        r"""Window the input.

        We take an input, either a 4d tensor or a dictionary of 4d
        tensors, and return a windowed version of it. If it's a 4d
        tensor, we return a 5d tensor, with windows indexed along the
        3rd dimension. If it's a dictionary, we return a dictionary with
        the same keys and have changed all the values to 5d tensors,
        with windows indexed along the 3rd dimension.

        If it's a 4d tensor, we use the ``idx`` entry in the ``windows``
        list. If it's a dictionary, we assume it's keys are ``(scale,
        orientation)`` tuples and so use ``windows[key[0]]`` to find the
        appropriately-sized window (this is the case for, e.g., the
        steerable pyramid). If we want to use differently-structured
        dictionaries, we'll need to restructure this.

        Parameters
        ----------
        x
            Either a 4d tensor or a dictionary of 4d tensors
        idx
            Which entry in the ``windows`` list to use. Only used if
            ``x`` is a tensor

        Returns
        -------
        windowed_x
            Same type as ``x``, see above for how it's created

        Raises
        ------
        ValueError
            If ``x`` is not 4d tensor or dictionary of 4d tensors

        See Also
        --------
        pool : pool the windowed input (get the weighted average)
        forward : perform the windowing and pooling simultaneously

        """
        if isinstance(x, dict):
            if list(x.values())[0].ndimension() != 4:
                raise ValueError(
                    "PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                    " Unsqueeze until this is true!"
                )
            # one way to make this more general: figure out the size of
            # the tensors in x and in self.windows, and intelligently
            # lookup which should be used.
            return dict(
                (
                    k,
                    oe.contract(
                        "bchw,ahw,ehw->bceahw",
                        v,
                        self.angle_windows[k[0]],
                        self.ecc_windows[k[0]],
                        backend="torch",
                    ).flatten(2, 3),
                )
                for k, v in x.items()
            )
        else:
            if x.ndimension() != 4:
                raise ValueError(
                    "PoolingWindows input must be 4d tensors or a dict of 4d tensors!"
                    " Unsqueeze until this is true!"
                )
            return oe.contract(
                "bchw,ahw,ehw->bceahw",
                x,
                self.angle_windows[idx],
                self.ecc_windows[idx],
                backend="torch",
            ).flatten(2, 3)

    def pool(
        self, windowed_x: dict[torch.Tensor] | torch.Tensor, idx: int = 0
    ) -> dict[torch.Tensor] | torch.Tensor:
        r"""Pool the windowed input.

        We take the windowed input (as returned by :meth:`window`)
        and perform a weighted average, dividing each windowed statistic
        by the sum of the window that generated it.

        The input must either be a 5d tensor or a dictionary of 5d
        tensors and we collapse across the spatial dimensions, returning
        a 3d tensor or a dictionary of 3d tensors.

        Similar to :meth:`window`, if it's a tensor, we use the
        ``idx`` entry in the ``windows`` list. If it's a dictionary, we
        assume it's keys are ``(scale, orientation)`` tuples and so use
        ``windows[key[0]]`` to find the appropriately-sized window (this
        is the case for, e.g., the steerable pyramid). If we want to use
        differently-structured dictionaries, we'll need to restructure
        this

        Parameters
        ----------
        windowed_x
            Either a 5d tensor or a dictionary of 5d tensors
        idx
            Which entry in the ``windows`` list to use. Only used if
            ``windowed_x`` is a tensor

        Returns
        -------
        pooled_x
            Same type as ``windowed_x``, see above for how it's created.

        See Also
        --------
        window : window the input
        forward : perform the windowing and pooling simultaneously

        """
        if isinstance(windowed_x, dict):
            # one way to make this more general: figure out the size
            # of the tensors in x and in self.angle_windows, and
            # intelligently lookup which should be used.
            return dict((k, v.sum((-1, -2))) for k, v in windowed_x.items())
        else:
            return windowed_x.sum((-1, -2))

    def project(
        self, pooled_x: dict[torch.Tensor] | torch.Tensor, idx: int = 0
    ) -> dict[torch.Tensor] | torch.Tensor:
        r"""Project pooled values back onto an image.

        For visualization purposes, you may want to project the pooled
        values (or values that have been pooled and then transformed in
        other ways) back onto an image. This method will do that for
        you.

        It takes a 3d tensor or dictionary of 3d tensors (like the
        output of :meth:`forward` / :meth:`pool`; the final dimension must
        have a value for each window) and returns a 4d tensor or
        dictionary of 4d tensors (like the input of :meth:`forward` /
        :meth:`window`).

        For example, if we have 100 windows, you must pass a i x j x 100
        tensor. For each of the i batches and j channels, we'll then
        multiply each of the 100 values by the corresponding window to
        end up with an i x j x 100 x height x width tensor. We then sum
        across windows to get i x j x heigth x width and return that.

        Parameters
        ----------
        pooled_x
            3d Tensor or a dictionary of 3d tensors
        idx
            Which entry in the ``windows`` list to use. Only used if
            ``pooled_x`` is a tensor.

        Returns
        -------
        x
            4d tensor or dictionary of 4d tensors

        Raises
        ------
        ValueError
            If ``pooled_x`` is not 3d tensor or dictionary of 3d tensors

        See Also
        --------
        forward : the opposite of this, going from image to pooled
            values

        """
        if isinstance(pooled_x, dict):
            if list(pooled_x.values())[0].ndimension() != 3:
                raise ValueError(
                    "PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                    " Squeeze until this is true!"
                )
            tmp = {}
            for k, v in pooled_x.items():
                # if keys are (scale, orientation) tuples, we want scale index
                # otherwise if key is a string, probably "mean_luminance" and
                # this corresponds to the lowest/largest scale
                window_key = k[0] if isinstance(k, tuple) else 0
                v = v.reshape(
                    (
                        *v.shape[:2],
                        self.ecc_windows[window_key].shape[0],
                        self.angle_windows[window_key].shape[0],
                    )
                )
                tmp[k] = oe.contract(
                    "bcea,ahw,ehw->bchw",
                    v,
                    self.angle_windows[window_key],
                    self.ecc_windows[window_key] / self.norm_factor[window_key],
                    backend="torch",
                )
            return tmp
        else:
            if pooled_x.ndimension() != 3:
                raise ValueError(
                    "PoolingWindows input must be 3d tensors or a dict of 3d tensors!"
                    " Squeeze until this is true!"
                )
            pooled_x = pooled_x.reshape(
                (
                    *pooled_x.shape[:2],
                    self.ecc_windows[idx].shape[0],
                    self.n_polar_windows,
                )
            )
            return oe.contract(
                "bcea,ahw,ehw->bchw",
                pooled_x,
                self.angle_windows[idx],
                self.ecc_windows[idx] / self.norm_factor[idx],
                backend="torch",
            )

    def save(self, save_path: str):
        r"""Save pooling windows model parameters.

        This function saves all necessary data for model initialization at the
        specified path. It does not save the window tensors themselves; these
        are saved during object initialization if the :attr:`cache_dir` argument was
        set.

        Parameters
        ----------
        save_path
            The file path you wish to save the model parameters to

        See Also
        --------
        load
            Method to load in the saved pooling windows parameters

        Examples
        --------
        To use, just input a file path in order to save the parameters needed
        for initializing the pooling window model.

        >>> import fenestration as fen
        >>> pw = fen.PoolingWindows(0.5, (256, 256))
        >>> pw.save("model_params.pt")
        >>> pw_new = fen.PoolingWindows.load("model_params.pt")

        """
        save_dict = {
            "scaling": self.scaling,
            "img_res": self.img_res,
            "min_ecc": self.min_ecc,
            "max_ecc": self.max_ecc,
            "num_scales": self.num_scales,
            "cache_dir": self.cache_dir,
            "window_type": self.window_type,
        }

        torch.save(save_dict, save_path)

    @classmethod
    def load(
        cls, load_path: str, cache_dir: str | None = None, **kwargs: Any
    ) -> nn.Module:
        r"""Load pooling windows parameters and initialize model.

        Helper function that can load the necessary data for model and output
        model instatiation with those parameters.

        Parameters
        ----------
        load_path
            The path to the file you wish to load
        cache_dir
            Optional path to a new cache directory to pass the model initialization,
            overriding the saved value. This allows you to e.g., load from a cache
            at a different location.
        kwargs
            Any additional kwargs to pass to :func:`torch.load`

        Returns
        -------
        pw
            A PoolingWindows object created with parameters from loaded dictionary.

        See Also
        --------
        save
            Method to save pooling windows parameters.

        Examples
        --------
        To use, just input a path to the file saved using :meth:`save` in order to load
        the parameters needed for initializing the pooling window model.

        >>> import fenestration as fen
        >>> pw = fen.PoolingWindows(0.5, (256, 256))
        >>> pw.save("model_params.pt")
        >>> pw_new = fen.PoolingWindows.load("model_params.pt")
        >>> pw_new
        PoolingWindows()

        """
        load_model = torch.load(load_path, weights_only=True, **kwargs)

        if cache_dir is not None:
            load_model["cache_dir"] = cache_dir

        pw = cls(**load_model)

        return pw

    def plot_windows(
        self,
        ax: plt.Axes | None = None,
        contour_levels: np.ndarray | int | None = None,
        colors: list[str] | str = "r",
        subset: bool = True,
        windows_scale: int = 0,
        **kwargs: Any,
    ) -> plt.Axes:
        r"""Plot the pooling windows on an image.

        This is just a simple little helper to plot the pooling windows
        on an axis. The intended use case is overlaying this on top of
        the image we're pooling.

        Any additional kwargs get passed to :meth:`~matplotlib.axes.Axes.contour`.

        Parameters
        ----------
        ax
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis.
        contour_levels
            The ``levels`` argument to pass to
            :meth:`~matplotlib.axes.Axes.contour`. From that
            documentation: "Determines the number and positions of the
            contour lines / regions. If an int ``n``, use ``n`` data
            intervals; i.e. draw ``n+1`` contour lines. The level
            heights are automatically chosen. If array-like, draw
            contour lines at the specified levels. The values must be in
            increasing order". If None, will plot the contour that gives
            the first intersection (.5 for raised-cosine windows,
            ``self.window_max_amplitude * np.exp(-.25/2)``, or half a standard
            deviation away from max, for gaussian windows), as this is
            the easiest to see.
        colors
            The ``colors`` argument to pass to
            :meth:`~matplotlib.axes.Axes.contour`. If a
            single character, all will have the same color; if a
            sequence, will cycle through the colors in ascending order
            (repeating if necessary).
        subset
            If True, will only plot four of the angle window
            slices. This is to save time and memory. If False, will plot
            all of them.
        windows_scale
            Which scale of the windows to use. ``windows`` is a list with
            different scales, so this specifies which one to use.

        Returns
        -------
        ax : :class:`~matplotlib.axes.Axes`
            The axis with the windows

        """
        if ax is None:
            fig = _plot._setup_fig(self.img_res)
            ax = fig.axes[0]
        if contour_levels is None:
            contour_levels = [self.window_intersecting_amplitude]
        # attempt to not have all the windows in memory at once...
        angle_windows = self.angle_windows[windows_scale]
        ecc_windows = self.ecc_windows[windows_scale] / self.norm_factor[windows_scale]
        if subset:
            angle_windows = angle_windows[:4]
        for a in angle_windows:
            windows = torch.einsum("hw,ehw->ehw", [a, ecc_windows])
            for w in windows:
                try:
                    # if this isn't true, then this window will be
                    # plotted weird
                    if not (w > contour_levels[0]).any():
                        continue
                except TypeError:
                    # in this case, it's an int
                    pass
                ax.contour(
                    _tensors._to_numpy(w), contour_levels, colors=colors, **kwargs
                )
        return ax

    def plot_window_values(
        self,
        im: torch.Tensor | None = None,
        ax: plt.Axes | None = None,
        subset: bool = True,
        windows_scale: int = 0,
        **kwargs: Any,
    ) -> plt.Axes:
        r"""Plot the windowed average values.

        This plots the average values of an image, as computed by these
        windows, and plots them using contourf as an RGB triple. We plot these
        within the window contours using :attr:`window_intersecting_amplitude`, so
        that if you call :meth:`plot_windows` with ``contour_levels=None``, they will
        outline these regions.

        Any additional kwargs are passed to :meth:`~matplotlib.axes.Axes.contourf`.

        Parameters
        ----------
        im
            The image whose average values we plot within the windows. If None,
            we plot random gray values instead.
        ax
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis.
        subset
            If True, will only plot four of the angle window
            slices. This is to save time and memory. If False, will plot
            all of them.
        windows_scale
            Which scale of the windows to use. ``windows`` is a list with
            different scales, so this specifies which one to use.

        Returns
        -------
        ax : :class:`~matplotlib.axes.Axes`
            The axis with the windows

        Raises
        ------
        ValueError
            If ``im`` has more than one batch or channel

        """
        if ax is None:
            fig = _plot._setup_fig(self.img_res)
            ax = fig.axes[0]
        contour_level = self.window_intersecting_amplitude
        # attempt to not have all the windows in memory at once...
        angle_windows = self.angle_windows[windows_scale]
        ecc_windows = self.ecc_windows[windows_scale] / self.norm_factor[windows_scale]
        if im is not None:
            im = im.squeeze()
            if im.ndim > 2:
                raise ValueError("im can only have one batch and channel!")
        if subset:
            angle_windows = angle_windows[:4]
        for a in angle_windows:
            windows = torch.einsum("hw,ehw->ehw", [a, ecc_windows])
            if im is not None:
                # if we use the windows to generate the color, the most obvious
                # thing is that windows near the periphery have most of their mass
                # off the image, and so windows get darker near the edge fo the
                # image. this corrects for that
                norm_windows = torch.einsum(
                    "hw,hw,ehw->e",
                    [torch.ones_like(im), a, self.ecc_windows[windows_scale]],
                )
                output = torch.einsum(
                    "hw,hw,ehw->e", [im, a, self.ecc_windows[windows_scale]]
                )
                colors = _tensors._to_numpy(output / norm_windows)
            else:
                colors = np.random.rand(windows.shape[0])
            # and convert into grey RGB triples
            colors = [(c, c, c) for c in colors]
            for i, w in enumerate(windows):
                try:
                    # if this isn't true, then this window will be
                    # plotted weird
                    if not (w > contour_level).any():
                        continue
                except TypeError:
                    # in this case, it's an int
                    pass
                ax.contourf(
                    _tensors._to_numpy(w),
                    [contour_level, 1],
                    colors=[colors[i]],
                    **kwargs,
                )
        return ax

    def plot_window_widths(
        self,
        units: Literal["degrees", "pixels"] = "degrees",
        scale_num: int = 0,
        figsize: tuple[int, int] = (5, 5),
        jitter: float | None = 0.25,
        ax: plt.Axes | None = None,
    ) -> Figure:
        r"""Plot the widths of the windows, in degrees or pixels.

        We plot the width of the window in both angular and radial
        direction, as well as showing the 'top', 'half', and 'full'
        widths (top is the width of the flat-top region of each window,
        where the window's value is 1; full is the width of the entire
        window; half is the width at the half-max value, which is what
        corresponds to the scaling value)

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        If the unit is 'pixels', then we also need to know which
        ``scale_num`` to plot (the windows are created at different
        scales, and so come in different pixel sizes)

        Parameters
        ----------
        units
            Whether to show the information in degrees or pixels (both
            the width and the window location will be presented in the
            same unit).
        scale_num
            Which scale window we should plot
        figsize
            The size of the figure to create
        jitter
            Whether to add a little bit of jitter to the x-axis to
            separate the radial and angular widths. There are only two
            values we separate, so we don't add actual jitter, just move
            one up by the value specified by jitter, the other down by
            that much (we use the same value at each eccentricity).
        ax
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis.

        Returns
        -------
        fig : :class:`~matplotlib.figure.Figure`
            The figure containing the plot

        Raises
        ------
        ValueError
            If ``units`` are not 'pixels' or 'degrees'

        """
        if units == "degrees":
            data = self._window_width_degrees
            central_ecc = self._central_eccentricity_degrees
        elif units == "pixels":
            data = self.window_width_pixels[scale_num]
            central_ecc = self.central_eccentricity_pixels[scale_num]
        else:
            raise ValueError(
                f"units must be one of {'pixels', 'degrees'}, not {units}!"
            )
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        if jitter is not None:
            jitter_vals = {"radial": -jitter, "angular": jitter}
        else:
            jitter_vals = {"radial": 0, "angular": 0}
        colors = {"radial": "C0", "angular": "C1"}
        sizes = {"full": 5, "half": 10, "top": 15}
        for direc, height in itertools.product(
            ["radial", "angular"], ["top", "half", "full"]
        ):
            m, _, _ = ax.stem(
                central_ecc + jitter_vals[direc],
                data[direc + "_" + height],
                linefmt=colors[direc],
                markerfmt=colors[direc] + ".",
                label=direc + "_" + height,
            )
            m.set(markersize=sizes[height])
        ax.set_ylabel(f"Window width ({units})")
        ax.set_xlabel(f"Window central eccentricity ({units})")
        ax.legend(loc="upper left")
        return fig

    def plot_window_areas(
        self,
        units: Literal["degrees", "pixels"] = "degrees",
        scale_num: int = 0,
        figsize: tuple[int, int] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> Figure:
        r"""Plot the approximate areas of the windows, in degrees or pixels.

        We plot the approximate area of the window, calculated using
        'top', 'half', and 'full' widths (top is the width of the
        flat-top region of each window, where the window's value is 1;
        full is the width of the entire window; half is the width at the
        half-max value, which is what corresponds to the scaling
        value). To get the approximate area, we multiply the radial
        width against the corresponding angular width, then divide by
        :math:`\frac{\pi}{4}`.

        The half area shown here is what we use to compare against a
        threshold value to determine the minimal eccentricity at which
        windows contain more than 1 pixel.

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity.

        If the unit is 'pixels', then we also need to know which
        ``scale_num`` to plot (the windows are created at different
        scales, and so come in different pixel sizes).

        Parameters
        ----------
        units
            Whether to show the information in degrees or pixels (both
            the area and the window location will be presented in the
            same unit).
        scale_num
            Which scale window we should plot
        figsize
            The size of the figure to create
        ax
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis

        Returns
        -------
        fig : :class:`~matplotlib.figure.Figure`
            The figure containing the plot

        Raises
        ------
        ValueError
            If ``units`` are not 'pixels' or 'degrees'

        """
        if units == "degrees":
            data = self._window_approx_area_degrees
            central_ecc = self._central_eccentricity_degrees
        elif units == "pixels":
            data = self.window_approx_area_pixels[scale_num]
            central_ecc = self.central_eccentricity_pixels[scale_num]
        else:
            raise ValueError(
                f"units must be one of {'pixels', 'degrees'}, not {units}!"
            )
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        sizes = {"full": 5, "half": 10, "top": 15}
        for height in ["top", "half", "full"]:
            m, _, _ = ax.stem(
                central_ecc, data[height], linefmt="C0", markerfmt="C0.", label=height
            )
            m.set(markersize=sizes[height])
        ax.set_ylabel(f"Window area ({units})")
        ax.set_xlabel(f"Window central eccentricity ({units})")
        ax.legend(loc="upper left")
        return fig

    def plot_window_checks(
        self, angle_n: int | list[int] = 0, scale: int = 0
    ) -> Figure:
        r"""Make some plots to check whether windows have been normalized properly.

        This creates a figure with two sets of plots: the first row shows the
        L1-norm of the windows, the second shows the sum. Each row will have
        one plot and, if everything worked correctly, they should each look
        like a sigmoid function that runs from 1 for small eccentricities to 0
        for high eccentricities

        You can plot multiple angle slices, and each should look more or
        less the same

        Parameters
        ----------
        angle_n : int or list, optional
            Which angle slice(s) to show. Can be a single int or a list
            of ints, in which case we plot each as a separate color
        scale : int, optional
            We plot this for one scale at a time. this specifies the
            scale.

        Returns
        -------
        fig : :class:`~matplotlib.figure.Figure`
            The figure containing the plot

        """
        if not hasattr(angle_n, "__iter__"):
            angle_n = [angle_n]
        einsum_str = "ahw,ehw->eahw"
        legend = True
        funcs = [lambda x: torch.norm(x, 1, (-1, -2)), lambda x: torch.sum(x, (-1, -2))]
        angle_all = self.angle_windows[scale].shape[0]
        windows = torch.einsum(
            einsum_str, self.angle_windows[scale][angle_n], self.ecc_windows[scale]
        )
        fig, axes = plt.subplots(2, 1, figsize=(5, 10), gridspec_kw={"hspace": 0.4})
        for i, (f, name) in enumerate(zip(funcs, ["L1-norm", "Sum"])):
            d = f(windows).numpy()
            # most of the time, self._central_eccentricity_degrees
            # and d will be same size, but sometimes they will not
            # not. this happens because _central_eccentricity_degrees
            # contains all windows that we constructed, but the
            # ecc_windows dictionary throws away any windows that
            # have all zero (or close to zero) values. this will be
            # those at the end, because they're off the image
            ecc = self._central_eccentricity_degrees[: d.shape[0]]
            axes[i].semilogx(ecc, d)
            for j, dj in enumerate(d.transpose(1, 0)):
                label = angle_n[j] if i == 0 else None
                axes[i].scatter(ecc, dj, label=label)
            axes[i].set(
                title="Windows", xlabel="Window central eccentricity (deg)", ylabel=name
            )
            fig.text(
                0.5,
                [0.91, 0.47][i],
                ha="center",
                fontsize=1.5 * plt.rcParams["font.size"],
                s=f"{name} of windows in some angle slices out of {angle_all}",
            )
        if legend:
            fig.legend(loc="center right", title="Angle slices")
        return fig

    def summarize_window_sizes(
        self, units: Literal["pixels", "degrees"] = "pixels"
    ) -> dict:
        r"""Summarize window sizes.

        This function returns a dictionary summarizing the window sizes
        at the minimum and maximum eccentricity in the specified units.
        The ``"min_window"`` and ``"max_window"`` are those whose centers
        are closest to :attr:`min_ecc` and :attr:`max_ecc`,
        respectively. For both of the window sizes, we return a dictionary
        containing the center, full-width half-max (FWHM, in the radial
        direction), and approximate area (at half-max). If ``units="pixels"``,
        we calculate these separately for each scale.

        Parameters
        ----------
        units
            Which units to return the window size summary in

        Returns
        -------
        sizes
            Dictionary with the keys described above, summarizing window
            sizes. All values are scalar floats.

        Raises
        ------
        ValueError
            If ``units`` are not "pixels" or "degrees"

        Examples
        --------
        In order to display the window size parameters nicely,
        :func:`~pprint.pprint` is recommended:

        >>> from pprint import pprint
        >>> import fenestration as fen
        >>> pw = fen.PoolingWindows(0.5, (256, 256))
        >>> summary = pw.summarize_window_sizes()
        >>> pprint(summary)
        {'max_window_scale_0_area': np.float64(1489.7697961809874),
        'max_window_scale_0_center': np.float64(123.18551268877933),
        'max_window_scale_0_fwhm': np.float64(61.59275634438966),
        'min_window_scale_0_area': np.float64(2.721047914586897),
        'min_window_scale_0_center': np.float64(5.26463355455719),
        'min_window_scale_0_fwhm': np.float64(2.632316777278595)}

        """
        min_idx = np.abs(self._central_eccentricity_degrees - self.min_ecc).argmin()
        max_idx = np.abs(self._central_eccentricity_degrees - self.max_ecc).argmin()
        sizes = {}

        if units == "degrees":
            central_ecc = [self._central_eccentricity_degrees]
            widths = [self._window_width_degrees]
            areas = [self._window_approx_area_degrees]
        elif units == "pixels":
            central_ecc = self.central_eccentricity_pixels
            widths = self.window_width_pixels
            areas = self.window_approx_area_pixels
        else:
            raise ValueError(
                f"units must be one of {'pixels', 'degrees'}, not {units}!"
            )

        for i in range(len(central_ecc)):
            for extrem, idx in zip(["min", "max"], [min_idx, max_idx]):
                scale_idx = f"_scale_{i}_" if units == "pixels" else "_"
                sizes[f"{extrem}_window{scale_idx}center"] = central_ecc[i][idx]
                sizes[f"{extrem}_window{scale_idx}fwhm"] = widths[i]["radial_half"][idx]
                sizes[f"{extrem}_window{scale_idx}area"] = areas[i]["half"][idx]

        return sizes
