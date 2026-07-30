#!/usr/bin/env python3
import matplotlib as mpl
import numpy as np
import pytest
import torch

import fenestration as fen

# use the html backend, so we don't need to have ffmpeg
mpl.rcParams["animation.writer"] = "html"
# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
mpl.use("agg")


def get_x(x_eval):
    if x_eval == "sym":
        return torch.linspace(-5, 5, 101)
    elif x_eval == "asym":
        return torch.linspace(-5, 6, 101)


class TestSampling:
    @pytest.mark.parametrize("val_samp", [0.5, 1, 2])
    @pytest.mark.parametrize("x_eval", ["sym", "asym"])
    def test_check_val_sampling(self, val_samp, x_eval):
        fen.sampling.check_sampling(
            val_sampling=val_samp,
            pix_sampling=None,
            func=fen.pooling.gaussian,
            x=get_x(x_eval),
        )

    @pytest.mark.parametrize("pix_samp", [2, 10, 25])
    @pytest.mark.parametrize("x_eval", ["sym", "asym"])
    def test_check_pix_sampling(self, pix_samp, x_eval):
        fen.sampling.check_sampling(
            val_sampling=None,
            pix_sampling=pix_samp,
            func=fen.pooling.raised_cosine,
            x=get_x(x_eval),
        )

    @pytest.mark.parametrize("x_eval", ["sym", "asym"])
    def test_check_sampling_error(self, x_eval):
        with pytest.raises(
            Exception, match="One of val_sampling or pix_sampling must be None!"
        ):
            fen.sampling.check_sampling(
                val_sampling=2,
                pix_sampling=2,
                func=fen.pooling.raised_cosine,
                x=get_x(x_eval),
            )

    @pytest.mark.parametrize("x_eval", ["sym", "asym"])
    def test_check_small_residuals(self, x_eval):
        _, _, _, _, residuals = fen.sampling.check_sampling(0.5, x=get_x(x_eval))
        assert np.allclose(residuals, 0)

    def test_check_interp_fun(self):
        # only checking symmetric X because using 0 as center point
        # of functions
        x_eval = get_x("sym")
        _, _, interps, _, _ = fen.sampling.check_sampling(
            0.5, func=fen.pooling.gaussian, x=x_eval
        )
        orig_fun = fen.pooling.gaussian(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert np.allclose(interps[:, cent], np.array(orig_fun))

    def test_check_bad_interp(self):
        # only checking symmetric X because using 0 as center point
        # of functions
        x_eval = get_x("sym")
        _, _, interps, _, _ = fen.sampling.check_sampling(
            2, func=fen.pooling.gaussian, x=x_eval
        )
        orig_fun = fen.pooling.gaussian(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert not np.allclose(interps[:, cent], np.array(orig_fun))

    def test_check_bad_interp_avgdiff(self):
        # only checking symmetric X because using 0 as center point
        # of functions
        x_eval = get_x("sym")
        _, _, interps, _, _ = fen.sampling.check_sampling(
            2, func=fen.pooling.gaussian, x=x_eval
        )
        orig_fun = fen.pooling.gaussian(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert np.mean(np.abs(interps[:, cent] - np.array(orig_fun))) > 0.001

    def test_check_interp_crossfunction(self):
        # only checking symmetric X because using 0 as center point
        # of functions
        x_eval = get_x("sym")
        _, _, interps, _, _ = fen.sampling.check_sampling(
            0.5, func=fen.pooling.gaussian, x=x_eval
        )
        orig_fun = fen.pooling.raised_cosine(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert not np.allclose(interps[:, cent], np.array(orig_fun))

    @pytest.mark.parametrize("x_eval", ["sym", "asym"])
    @pytest.mark.parametrize("check_idx", [1, 50, 100])
    def test_check_interp_order(self, x_eval, check_idx):
        _, _, interps, _, _ = fen.sampling.check_sampling(0.5, x=get_x(x_eval))
        max_idx = np.argmax(interps[check_idx])
        assert check_idx == max_idx

    def test_check_residuals(self):
        # only testing asymmetric x because symmetric x will not recognize
        # flipped function
        x_eval = get_x("asym")
        _, full, interps, _, residuals = fen.sampling.check_sampling(0.5, x=x_eval)
        assert np.allclose(residuals, np.pow(full - interps, 2).sum(0))
        assert not np.allclose(residuals, np.pow(full[::-1] - interps, 2).sum(0))
