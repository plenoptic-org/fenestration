#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import plenoptic as po
import pytest

import fenestration as fen

# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
mpl.use("agg")


# following https://github.com/scverse/scanpy/issues/1662
@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


class TestPlotting:
    def test_plotting_windows(self, pool_win):
        pool_win.plot_windows()

    @pytest.mark.parametrize("contour", [None, 0, 1, [0, 2]])
    @pytest.mark.parametrize("color", ["b", "r"])
    @pytest.mark.parametrize("subset", [False, True])
    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_windows_args(self, pool_win, contour, color, subset, win_scale):
        pool_win.plot_windows(
            contour_levels=contour, colors=color, subset=subset, windows_scale=win_scale
        )

    @pytest.mark.parametrize("contour", [None, 0, 1, [0, 2]])
    @pytest.mark.parametrize("color", ["b", "r"])
    @pytest.mark.parametrize("subset", [False, True])
    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_windows_axargs(self, pool_win, contour, color, subset, win_scale):
        _, axes = plt.subplots(1, 1, figsize=(4, 4))
        pool_win.plot_windows(
            ax=axes,
            contour_levels=contour,
            colors=color,
            subset=subset,
            windows_scale=win_scale,
        )

    def test_plotting_window_values(self, pool_win):
        pool_win.plot_window_values()

    @pytest.mark.parametrize("subset", [False, True])
    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_window_values_args(self, pool_win, subset, win_scale):
        pool_win.plot_window_values(im=None, subset=subset, windows_scale=win_scale)

    @pytest.mark.parametrize("subset", [False, True])
    def test_plotting_window_values_imgargs(self, pool_win, rand_img, subset):
        _, axes = plt.subplots(1, 1, figsize=(4, 4))
        pool_win.plot_window_values(
            im=rand_img, ax=axes, subset=subset, windows_scale=0
        )

    def test_plotting_window_values_err(self, pool_win, rand_img):
        with pytest.raises(ValueError, match="Size of label 'h'"):
            pool_win.plot_window_values(im=rand_img, windows_scale=1)

    def test_plotting_window_widths(self, pool_win):
        pool_win.plot_window_widths()

    @pytest.mark.parametrize("units", ["pixels", "degrees"])
    @pytest.mark.parametrize("scale_num", [0, 1])
    @pytest.mark.parametrize("figsize", [(4, 4), (5, 5)])
    @pytest.mark.parametrize("jitter", [None, 0, 0.25])
    def test_plotting_window_widths_args(
        self, pool_win, units, scale_num, figsize, jitter
    ):
        pool_win.plot_window_widths(
            units=units, scale_num=scale_num, figsize=figsize, jitter=jitter
        )

    @pytest.mark.parametrize("units", ["pixels", "degrees"])
    @pytest.mark.parametrize("scale_num", [0, 1])
    @pytest.mark.parametrize("figsize", [(4, 4), (5, 5)])
    @pytest.mark.parametrize("jitter", [None, 0, 0.25])
    def test_plotting_window_widths_axargs(
        self, pool_win, units, scale_num, figsize, jitter
    ):
        _, axes = plt.subplots(1, 1)
        pool_win.plot_window_widths(
            units=units, scale_num=scale_num, jitter=jitter, figsize=figsize, ax=axes
        )

    def test_plotting_window_areas(self, pool_win):
        pool_win.plot_window_areas()

    @pytest.mark.parametrize("units", ["pixels", "degrees"])
    @pytest.mark.parametrize("scale_num", [0, 1])
    @pytest.mark.parametrize("figsize", [(4, 4), (5, 5)])
    def test_plotting_window_areas_args(self, pool_win, units, scale_num, figsize):
        pool_win.plot_window_areas(units=units, scale_num=scale_num, figsize=figsize)

    @pytest.mark.parametrize("units", ["pixels", "degrees"])
    @pytest.mark.parametrize("scale_num", [0, 1])
    @pytest.mark.parametrize("figsize", [(4, 4), (5, 5)])
    def test_plotting_window_areas_axargs(self, pool_win, units, scale_num, figsize):
        _, axes = plt.subplots(1, 1, figsize=(4, 4))
        pool_win.plot_window_areas(
            units=units, scale_num=scale_num, figsize=figsize, ax=axes
        )

    def test_po_tensor_plot(self):
        angle_w, ecc_w = fen.pooling.create_pooling_windows(0.87, (256, 256))
        po.plot.imshow(ecc_w.unsqueeze(0))
        po.plot.imshow(angle_w.unsqueeze(0))

    def test_plot_window_checks(self, pool_win):
        pool_win.plot_window_checks()

    @pytest.mark.parametrize("angle_n", [0, 4])
    @pytest.mark.parametrize("scale", [0, 1])
    def test_plot_window_checks_args(self, pool_win, angle_n, scale):
        pool_win.plot_window_checks(angle_n, scale)

    @pytest.mark.parametrize("scale", [0, 1])
    def test_plot_window_checks_arglist(self, pool_win, scale):
        pool_win.plot_window_checks([0, 4], scale)
