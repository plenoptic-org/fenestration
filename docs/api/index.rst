.. _api:

API
===

Pooling Windows object
----------------------

The main object you will interact with is :class:`~fenestration.PoolingWindows` which will create
and plot the generated windows. Additionally, :meth:`~fenestration.create_pooling_windows` will return
the angle and eccentricity windows without the added functionality of :class:`~fenestration.PoolingWindows`.

.. currentmodule:: fenestration
.. autosummary::
    :toctree: generated
    :signatures: none
    :template: pw_module.rst.jinja

    PoolingWindows

    :toctree: generated
    :signatures: none

    create_pooling_windows

Pooling
-------

These are additional helper functions related to creating pooling windows. The two
window types are :meth:`~fenestration.pooling.gaussian` and :meth:`~fenestration.pooling.raised_cosine`.
Additionally, :meth:`~fenestration.pooling.normalize_windows` is helpful for ensuring that varying
eccentricities of :meth:`~fenestration.create_pooling_windows` all contribute equally.

.. currentmodule:: fenestration.pooling
.. autosummary::
    :toctree: generated
    :signatures: none

    gaussian
    raised_cosine
    normalize_windows

Calculate
---------

Methods for computing conversions, from degrees to pixels (:meth:`~fenestration.calculate.deg_to_pix`) and
calculating a specific scaling value based on a required number of windows and eccentricity ranges
(:meth:`~fenestration.calculate.scaling`).

.. currentmodule:: fenestration.calculate
.. autosummary::
    :toctree: generated
    :signatures: none

    scaling
    deg_to_pix

Sampling
--------

When creating pooling windows, it is important to ensure that any sampling does not result in aliasing,
in which there are incorrect measurements in the reconstructed signal. These methods allow you to check this
property using a specified function, domain, and how to sample that domain. Furthermore, you can
visualize interpolation coefficients and resulting signals.

.. currentmodule:: fenestration.sampling
.. autosummary::
    :toctree: generated
    :signatures: none

    check_sampling
    plot_coeffs
    interpolation_plot
    create_movie
