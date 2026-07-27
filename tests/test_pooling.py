#!/usr/bin/env python3
import os.path as op
import pathlib
import time

import numpy as np
import pytest
import torch

import pooling
from conftest import DEVICE


class TestPooling:
    def test_creation(self):
        pooling.pooling.create_pooling_windows(0.87, (256, 256))

    @pytest.mark.parametrize("region_width", [0.5, 0.7])
    def test_creation_args(self, region_width):
        pooling.pooling.create_pooling_windows(
            0.87, (100, 100), 0.2, 30, 1.2, transition_region_width=region_width
        )

    def test_creation_gaussian(self):
        pooling.pooling.create_pooling_windows(
            0.87, (100, 100), 0.2, 30, 1.2, "gaussian"
        )

    @pytest.mark.parametrize("n_windows", [4, 4.5])
    def test_ecc_nwindows(self, n_windows):
        pooling.pooling._log_eccentricity_windows((256, 256), n_windows=n_windows)

    @pytest.mark.parametrize("window_spacing", [0.5, 1])
    def test_ecc_window_spacing(self, window_spacing):
        pooling.pooling._log_eccentricity_windows(
            (256, 256), window_spacing=window_spacing
        )

    @pytest.mark.parametrize("res", [(256, 256), (1000, 1000), 100])
    def test_angle_windows(self, res):
        pooling.pooling._polar_angle_windows(10, res)

    def test_angle_windows_notint(self):
        with pytest.raises(Exception, match="n_windows must be an integer"):
            pooling.pooling._polar_angle_windows(1.5, (256, 256))

    def test_angle_windows_onewin(self):
        with pytest.raises(Exception, match="We cannot handle one window correctly!"):
            pooling.pooling._polar_angle_windows(1, (256, 256))

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert pooling.pooling.calculate._angular_window_spacing(2) == np.pi
        assert pooling.pooling.calculate._angular_n_windows(2) == np.pi
        with pytest.raises(
            Exception, match="Exactly one of n_windows or scaling must be set!"
        ):
            pooling.pooling.calculate._eccentricity_window_spacing()
        assert np.allclose(
            pooling.pooling.calculate._eccentricity_window_spacing(n_windows=4),
            0.8502993454155389,
        )
        assert np.allclose(
            pooling.pooling.calculate._eccentricity_window_spacing(scaling=0.87),
            0.8446653390527211,
        )
        assert np.allclose(
            pooling.pooling.calculate._eccentricity_window_spacing(5, 10, scaling=0.87),
            0.8446653390527211,
        )
        assert np.allclose(
            pooling.pooling.calculate._eccentricity_window_spacing(5, 10, n_windows=4),
            0.1732867951399864,
        )
        assert np.allclose(
            pooling.pooling.calculate._eccentricity_n_windows(0.8502993454155389), 4
        )
        assert np.allclose(
            pooling.pooling.calculate._eccentricity_n_windows(
                0.1732867951399864, 5, 10
            ),
            4,
        )
        assert np.allclose(pooling.pooling.calculate.scaling(4), 0.8761474337786708)
        assert np.allclose(
            pooling.pooling.calculate.scaling(4, 5, 10), 0.17350368946058647
        )
        assert np.isinf(pooling.pooling.calculate.scaling(4, 0))

    @pytest.mark.parametrize("num_scales", [1, 3])
    def test_PoolingWindows_cosine(self, rand_img, num_scales):
        pw = pooling.PoolingWindows(
            0.5,
            rand_img.shape[2:],
            num_scales=num_scales,
            window_type="cosine",
        )
        pw(rand_img)

    @pytest.mark.parametrize("num_scales", [1, 3])
    def test_PoolingWindows(self, rand_img, num_scales):
        pw = pooling.PoolingWindows(
            0.5,
            rand_img.shape[2:],
            num_scales=num_scales,
            window_type="gaussian",
        )
        pw(rand_img)

    def test_PoolingWindows_totype(self, pool_win):
        assert pool_win.angle_windows[0].dtype == torch.float32
        pool_win.to(torch.float64)
        assert pool_win.angle_windows[0].dtype == torch.float64
        pool_win.to(torch.float32)
        assert pool_win.angle_windows[0].dtype == torch.float32

    def test_PoolingWindows_toimg(self, pool_win, rand_img):
        assert pool_win.angle_windows[0].dtype == torch.float32
        pool_win.to(torch.float64)
        assert pool_win.angle_windows[0].dtype == torch.float64
        pool_win.to(rand_img)
        assert pool_win.angle_windows[0].dtype == rand_img.dtype

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    def test_PoolingWindows_todevice(self, pool_win):
        pool_win.to("cpu")
        assert pool_win.angle_windows[0].device == "cpu"
        pool_win.to("cuda")
        assert pool_win.angle_windows[0].device == "cuda"

    @pytest.mark.parametrize("offset", [0.2, 0.5])
    def test_PoolingWindows_merge(self, rand_img, pool_win, offset):
        other_pool_win = pooling.PoolingWindows(0.7, rand_img.shape[2:])
        pool_win.merge(other_pool_win, scale_offset=offset)
        assert np.allclose(
            pool_win.angle_windows[offset], other_pool_win.angle_windows[0]
        )

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_PoolingWindows_window(self, pool_win, rand_img, idx):
        if idx == 0:
            win = pool_win.window(rand_img, idx=idx)
            assert len(win.shape) == 5
        elif idx == 1:
            with pytest.raises(ValueError, match="Size of label 'h'"):
                pool_win.window(rand_img, idx=idx)
        elif idx == 2:
            with pytest.raises(KeyError, match=f"{idx}"):
                pool_win.window(rand_img, idx=idx)

    def test_PoolingWindows_pool(self, pool_win, rand_img):
        windowed_x = pool_win.window(rand_img)
        pool_win.pool(windowed_x)

    def test_PoolingWindows_project(self, rand_img):
        pw = pooling.PoolingWindows(0.5, rand_img.shape[2:])
        pooled = pw(rand_img)
        pw.project(pooled)
        pw = pooling.PoolingWindows(0.5, rand_img.shape[2:], num_scales=3)
        pooled = pw(rand_img)
        pw.project(pooled)

    @pytest.mark.parametrize(
        "sh", [(256, 128), (256, 127), (256, 125), (125, 125), (127, 125)]
    )
    def test_PoolingWindows_nonsquare(self, rand_img, sh):
        # test PoolingWindows with weirdly-shaped iamges
        tmp = rand_img[..., : sh[0], : sh[1]]
        pw = pooling.PoolingWindows(0.9, tmp.shape[-2:])
        pw(tmp)

    def test_PoolingWindows_caching(self, rand_img, tmp_path):
        # first time we save, second we load
        new_path = tmp_path / "test_dir"
        new_path.mkdir()
        start_time = time.perf_counter()
        pw = pooling.PoolingWindows(
            0.8, rand_img.shape[-2:], num_scales=2, cache_dir=new_path
        )
        tot_time_new = time.perf_counter() - start_time
        assert new_path.exists()
        for i in pw.cache_paths:
            assert pathlib.Path(i).exists()
            assert pathlib.Path(i).is_relative_to(new_path)
        start_time = time.perf_counter()
        pw = pooling.PoolingWindows(
            0.8, rand_img.shape[-2:], num_scales=2, cache_dir=new_path
        )
        tot_time_cache = time.perf_counter() - start_time
        for i in pw.cache_paths:
            assert pathlib.Path(i).exists()
            assert pathlib.Path(i).is_relative_to(new_path)
        assert tot_time_cache < tot_time_new

    def test_PoolingWindows_cache_dne(self, rand_img, tmp_path):
        tmp_path = op.join(tmp_path, "new_dir")
        with pytest.raises(FileNotFoundError, match="directory does not exist!"):
            pooling.PoolingWindows(
                0.8, rand_img.shape[-2:], num_scales=2, cache_dir=tmp_path
            )

    def test_PoolingWindows_sep(self, rand_img, pool_win):
        # test the window and pool function separate of the forward function
        pooled_x1 = pool_win(rand_img)
        pooled_x2 = pool_win.pool(pool_win.window(rand_img))
        assert np.allclose(pooled_x1, pooled_x2)

    @pytest.mark.parametrize("num_scales", [1, 3])
    @pytest.mark.parametrize("input_fmt", ["dict", "tensor"])
    def test_reweighting(self, num_scales, input_fmt):
        pw = pooling.PoolingWindows(0.5, (256, 256), num_scales=num_scales)
        im = {
            (i,): torch.rand((1, 1, 256 // 2**i, 256 // 2**i), dtype=torch.float32)
            for i in range(num_scales)
        }
        if input_fmt == "dict":
            pw(im, weights=torch.ones(num_scales, 1, 1, 1, 1))
        elif input_fmt == "tensor":
            for i in range(num_scales):
                pw(im[(i,)], idx=i, weights=torch.ones(num_scales, 1, 1, 1, 1))

    def test_PoolingWindows_summarize_gaussian(self, pool_win):
        sizes = pool_win.summarize_window_sizes()
        assert np.allclose(sizes["min_window_center_degrees"], 0.6169492446746707)
        assert np.allclose(sizes["min_window_fwhm_degrees"], 0.30847462233733536)
        assert np.allclose(sizes["min_window_area_degrees"], 0.037367906541873275)
        assert np.allclose(sizes["max_window_center_degrees"], 14.435802268216328)
        assert np.allclose(sizes["max_window_fwhm_degrees"], 7.217901134108164)
        assert np.allclose(sizes["max_window_area_degrees"], 20.458874764448375)
        assert np.allclose(sizes["min_window_scale_0_center_pixels"], 5.26463355455719)
        assert np.allclose(sizes["min_window_scale_0_fwhm_pixels"], 2.632316777278595)
        assert np.allclose(sizes["min_window_scale_0_area_pixels"], 2.721047914586897)
        assert np.allclose(
            sizes["max_window_scale_0_center_pixels"], 123.18551268877933
        )
        assert np.allclose(sizes["max_window_scale_0_fwhm_pixels"], 61.59275634438966)
        assert np.allclose(sizes["max_window_scale_0_area_pixels"], 1489.7697961809874)
        assert np.allclose(sizes["min_window_scale_1_center_pixels"], 2.632316777278595)
        assert np.allclose(sizes["min_window_scale_1_fwhm_pixels"], 1.3161583886392976)
        assert np.allclose(sizes["min_window_scale_1_area_pixels"], 0.6802619786467242)
        assert np.allclose(sizes["max_window_scale_1_center_pixels"], 61.59275634438966)
        assert np.allclose(sizes["max_window_scale_1_fwhm_pixels"], 30.79637817219483)
        assert np.allclose(sizes["max_window_scale_1_area_pixels"], 372.44244904524686)

    def test_PoolingWindows_summarize_cosine(self, rand_img):
        pw = pooling.PoolingWindows(
            0.5, rand_img.shape[2:], num_scales=2, window_type="cosine"
        )
        sizes = pw.summarize_window_sizes()
        assert np.allclose(sizes["min_window_center_degrees"], 0.8201941016011038)
        assert np.allclose(sizes["min_window_fwhm_degrees"], 0.4100970508005519)
        assert np.allclose(sizes["min_window_area_degrees"], 0.06604397097574137)
        assert np.allclose(sizes["max_window_center_degrees"], 15.980724561828527)
        assert np.allclose(sizes["max_window_fwhm_degrees"], 7.990362280914264)
        assert np.allclose(sizes["max_window_area_degrees"], 25.072222129865402)
        assert np.allclose(sizes["min_window_scale_0_center_pixels"], 6.998989666996086)
        assert np.allclose(sizes["min_window_scale_0_fwhm_pixels"], 3.499494833498043)
        assert np.allclose(sizes["min_window_scale_0_area_pixels"], 4.80917520207354)
        assert np.allclose(sizes["max_window_scale_0_center_pixels"], 136.3688495942701)
        assert np.allclose(sizes["max_window_scale_0_fwhm_pixels"], 68.18442479713505)
        assert np.allclose(sizes["max_window_scale_0_area_pixels"], 1825.7034994476212)
        assert np.allclose(sizes["min_window_scale_1_center_pixels"], 3.499494833498043)
        assert np.allclose(sizes["min_window_scale_1_fwhm_pixels"], 1.7497474167490215)
        assert np.allclose(sizes["min_window_scale_1_area_pixels"], 1.202293800518385)
        assert np.allclose(sizes["max_window_scale_1_center_pixels"], 68.18442479713505)
        assert np.allclose(sizes["max_window_scale_1_fwhm_pixels"], 34.092212398567526)
        assert np.allclose(sizes["max_window_scale_1_area_pixels"], 456.4258748619053)
