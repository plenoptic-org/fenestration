#!/usr/bin/env python3
import os.path as op
import pathlib
import time

import numpy as np
import pytest
import torch

import fenestration as fen
from conftest import DEVICE


def unpack_dict(pw):
    pw_dict = pw.__dict__
    new_dict = {}
    for k, v in pw_dict.items():
        if k.startswith(("_", "training")):
            continue
        else:
            # check if dictionary
            if isinstance(v, dict):
                new_dict.update({f"{k}_{k_}": v_ for k_, v_ in v.items()})
            # check if list of dictionaries and concat with new keys
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for num, d in enumerate(pw_dict[k]):
                    new_dict.update({f"{k}_{num}_{k_}": v_ for k_, v_ in d.items()})
            # check if list of numpy arrays and concat with new keys
            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                new_dict.update({f"{k}_{k_}": v_ for k_, v_ in enumerate(v)})
            # if none of the above, just copy
            else:
                new_dict[k] = v

    return new_dict


class TestPooling:
    def test_creation(self):
        fen.pooling.create_pooling_windows(0.87, (256, 256))

    @pytest.mark.parametrize("region_width", [0.5, 0.7])
    def test_creation_args(self, region_width):
        fen.pooling.create_pooling_windows(
            0.87, (100, 100), 0.2, 30, 1.2, transition_region_width=region_width
        )

    def test_creation_gaussian(self):
        fen.pooling.create_pooling_windows(0.87, (100, 100), 0.2, 30, 1.2, "gaussian")

    @pytest.mark.parametrize("n_windows", [4, 4.5])
    def test_ecc_nwindows(self, n_windows):
        fen.pooling._log_eccentricity_windows((256, 256), n_windows=n_windows)

    @pytest.mark.parametrize("window_spacing", [0.5, 1])
    def test_ecc_window_spacing(self, window_spacing):
        fen.pooling._log_eccentricity_windows((256, 256), window_spacing=window_spacing)

    @pytest.mark.parametrize("res", [(256, 256), (1000, 1000), 100])
    def test_angle_windows(self, res):
        fen.pooling._polar_angle_windows(10, res)

    def test_angle_windows_notint(self):
        with pytest.raises(Exception, match="n_windows must be an integer"):
            fen.pooling._polar_angle_windows(1.5, (256, 256))

    def test_angle_windows_onewin(self):
        with pytest.raises(Exception, match="We cannot handle one window correctly!"):
            fen.pooling._polar_angle_windows(1, (256, 256))

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert fen.pooling.calculate._angular_window_spacing(2) == np.pi
        assert fen.pooling.calculate._angular_n_windows(2) == np.pi
        with pytest.raises(
            Exception, match="Exactly one of n_windows or scaling must be set!"
        ):
            fen.pooling.calculate._eccentricity_window_spacing()
        assert np.allclose(
            fen.pooling.calculate._eccentricity_window_spacing(n_windows=4),
            0.8502993454155389,
        )
        assert np.allclose(
            fen.pooling.calculate._eccentricity_window_spacing(scaling=0.87),
            0.8446653390527211,
        )
        assert np.allclose(
            fen.pooling.calculate._eccentricity_window_spacing(5, 10, scaling=0.87),
            0.8446653390527211,
        )
        assert np.allclose(
            fen.pooling.calculate._eccentricity_window_spacing(5, 10, n_windows=4),
            0.1732867951399864,
        )
        assert np.allclose(
            fen.pooling.calculate._eccentricity_n_windows(0.8502993454155389), 4
        )
        assert np.allclose(
            fen.pooling.calculate._eccentricity_n_windows(0.1732867951399864, 5, 10),
            4,
        )
        assert np.allclose(fen.pooling.calculate.scaling(4), 0.8761474337786708)
        assert np.allclose(fen.pooling.calculate.scaling(4, 5, 10), 0.17350368946058647)
        assert np.isinf(fen.pooling.calculate.scaling(4, 0))

    @pytest.mark.parametrize("num_scales", [1, 3])
    def test_PoolingWindows_cosine(self, rand_img, num_scales):
        pw = fen.PoolingWindows(
            0.5,
            rand_img.shape[2:],
            num_scales=num_scales,
            window_type="cosine",
        )
        pw(rand_img)

    @pytest.mark.parametrize("num_scales", [1, 3])
    def test_PoolingWindows(self, rand_img, num_scales):
        pw = fen.PoolingWindows(
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
        assert pool_win.angle_windows[0].device.type == "cpu"
        pool_win.to("cuda")
        assert pool_win.angle_windows[0].device.type == "cuda"

    @pytest.mark.parametrize("offset", [0.2, 0.5])
    def test_PoolingWindows_merge(self, rand_img, pool_win, offset):
        other_pool_win = fen.PoolingWindows(0.7, rand_img.shape[2:])
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
        pw = fen.PoolingWindows(0.5, rand_img.shape[2:])
        pooled = pw(rand_img)
        pw.project(pooled)
        pw = fen.PoolingWindows(0.5, rand_img.shape[2:], num_scales=3)
        pooled = pw(rand_img)
        pw.project(pooled)

    @pytest.mark.parametrize(
        "sh", [(256, 128), (256, 127), (256, 125), (125, 125), (127, 125)]
    )
    def test_PoolingWindows_nonsquare(self, rand_img, sh):
        # test PoolingWindows with weirdly-shaped iamges
        tmp = rand_img[..., : sh[0], : sh[1]]
        pw = fen.PoolingWindows(0.9, tmp.shape[-2:])
        pw(tmp)

    def test_PoolingWindows_caching(self, rand_img, tmp_path):
        # first time we save, second we load
        new_path = tmp_path / "test_dir"
        new_path.mkdir()
        start_time = time.perf_counter()
        pw = fen.PoolingWindows(
            0.8, rand_img.shape[-2:], num_scales=2, cache_dir=new_path
        )
        tot_time_new = time.perf_counter() - start_time
        assert new_path.exists()
        for i in pw.cache_paths:
            assert pathlib.Path(i).exists()
            assert pathlib.Path(i).is_relative_to(new_path)
        start_time = time.perf_counter()
        pw = fen.PoolingWindows(
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
            fen.PoolingWindows(
                0.8, rand_img.shape[-2:], num_scales=2, cache_dir=tmp_path
            )

    def test_PoolingWindows_save(self, rand_img, tmp_path):
        pw = fen.PoolingWindows(0.8, rand_img.shape[-2:])
        pw.save(tmp_path / "model.pt")
        assert pathlib.Path(tmp_path / "model.pt").exists()
        assert pathlib.Path(tmp_path / "model.pt").is_file()

    def test_PoolingWindows_loadcache(self, rand_img, tmp_path):
        pw = fen.PoolingWindows(0.8, rand_img.shape[-2:], cache_dir=tmp_path)
        assert pathlib.Path(pw.cache_dir) == tmp_path
        pw.save(tmp_path / "model.pt")
        new_dir = tmp_path / "newdir"
        new_dir.mkdir()
        # move cached file to new path. only 1 for num_scales=1, could do multiple
        new_path = pathlib.Path(pw.cache_paths[0]).name
        pathlib.Path(pw.cache_paths[0]).rename(new_dir / new_path)
        # load using new path where cached file now lives
        pw_load = fen.PoolingWindows.load(tmp_path / "model.pt", cache_dir=new_dir)
        assert pathlib.Path(pw_load.cache_dir) == new_dir
        assert (new_dir / new_path).exists()
        assert not (tmp_path / new_path).exists()

    @pytest.mark.parametrize("scaling", [0.5, 1])
    @pytest.mark.parametrize("ecc", [[0.5, 10], [1, 15]])
    @pytest.mark.parametrize("num_scales", [1, 3])
    @pytest.mark.parametrize("window_type", ["gaussian", "cosine"])
    @pytest.mark.parametrize("file_type", [".pt", ".csv"])
    def test_PoolingWindows_saveload(
        self, scaling, rand_img, ecc, num_scales, window_type, tmp_path, file_type
    ):
        pw = fen.PoolingWindows(
            scaling,
            rand_img.shape[-2:],
            min_eccentricity=ecc[0],
            max_eccentricity=ecc[1],
            num_scales=num_scales,
            window_type=window_type,
        )
        pw_dict = unpack_dict(pw)

        pw.save(tmp_path / pathlib.Path(f"./model{file_type}"))
        pw_new = fen.PoolingWindows.load(tmp_path / pathlib.Path(f"./model{file_type}"))
        pw_new_dict = unpack_dict(pw_new)

        for k, v in pw_dict.items():
            if isinstance(pw_dict[k], (torch.Tensor, np.ndarray)):
                assert torch.allclose(
                    torch.as_tensor(pw_dict[k]), torch.as_tensor(pw_new_dict[k])
                )
            else:
                # otherwise check individual equivalence
                assert pw_dict[k] == pw_new_dict[k]

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    def test_PoolingWindows_saveload_device(self, pool_win, tmp_path):
        pool_win.to("cuda")
        pool_win.save(tmp_path / "model.pt")
        assert pool_win.angle_windows[0].device.type == "cuda"
        pw = fen.PoolingWindows.load(tmp_path / "model.pt")
        assert pw.angle_windows[0].device.type == "cpu"

    def test_PoolingWindows_nofile_load(self):
        with pytest.raises(FileNotFoundError, match="No such file or directory:"):
            fen.PoolingWindows.load("fake_file.pt")

    def test_PoolingWindows_sep(self, rand_img, pool_win):
        # test the window and pool function separate of the forward function
        pooled_x1 = pool_win(rand_img)
        pooled_x2 = pool_win.pool(pool_win.window(rand_img))
        assert np.allclose(pooled_x1, pooled_x2)

    @pytest.mark.parametrize("num_scales", [1, 3])
    @pytest.mark.parametrize("input_fmt", ["dict", "tensor"])
    def test_reweighting(self, num_scales, input_fmt):
        pw = fen.PoolingWindows(0.5, (256, 256), num_scales=num_scales)
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
        size_deg = pool_win.summarize_window_sizes(units="degrees")
        assert np.allclose(size_deg["min_window_center"], 0.6169492446746707)
        assert np.allclose(size_deg["min_window_fwhm"], 0.30847462233733536)
        assert np.allclose(size_deg["min_window_area"], 0.037367906541873275)
        assert np.allclose(size_deg["max_window_center"], 14.435802268216328)
        assert np.allclose(size_deg["max_window_fwhm"], 7.217901134108164)
        assert np.allclose(size_deg["max_window_area"], 20.458874764448375)

        size_pix = pool_win.summarize_window_sizes(units="pixels")
        assert np.allclose(size_pix["min_window_scale_0_center"], 5.26463355455719)
        assert np.allclose(size_pix["min_window_scale_0_fwhm"], 2.632316777278595)
        assert np.allclose(size_pix["min_window_scale_0_area"], 2.721047914586897)
        assert np.allclose(size_pix["max_window_scale_0_center"], 123.18551268877933)
        assert np.allclose(size_pix["max_window_scale_0_fwhm"], 61.59275634438966)
        assert np.allclose(size_pix["max_window_scale_0_area"], 1489.7697961809874)
        assert np.allclose(size_pix["min_window_scale_1_center"], 2.632316777278595)
        assert np.allclose(size_pix["min_window_scale_1_fwhm"], 1.3161583886392976)
        assert np.allclose(size_pix["min_window_scale_1_area"], 0.6802619786467242)
        assert np.allclose(size_pix["max_window_scale_1_center"], 61.59275634438966)
        assert np.allclose(size_pix["max_window_scale_1_fwhm"], 30.79637817219483)
        assert np.allclose(size_pix["max_window_scale_1_area"], 372.44244904524686)

    def test_PoolingWindows_summarize_cosine(self, rand_img):
        pw = fen.PoolingWindows(
            0.5, rand_img.shape[2:], num_scales=2, window_type="cosine"
        )
        sizes_deg = pw.summarize_window_sizes(units="degrees")
        assert np.allclose(sizes_deg["min_window_center"], 0.8201941016011038)
        assert np.allclose(sizes_deg["min_window_fwhm"], 0.4100970508005519)
        assert np.allclose(sizes_deg["min_window_area"], 0.06604397097574137)
        assert np.allclose(sizes_deg["max_window_center"], 15.980724561828527)
        assert np.allclose(sizes_deg["max_window_fwhm"], 7.990362280914264)
        assert np.allclose(sizes_deg["max_window_area"], 25.072222129865402)

        size_pix = pw.summarize_window_sizes(units="pixels")
        assert np.allclose(size_pix["min_window_scale_0_center"], 6.998989666996086)
        assert np.allclose(size_pix["min_window_scale_0_fwhm"], 3.499494833498043)
        assert np.allclose(size_pix["min_window_scale_0_area"], 4.80917520207354)
        assert np.allclose(size_pix["max_window_scale_0_center"], 136.3688495942701)
        assert np.allclose(size_pix["max_window_scale_0_fwhm"], 68.18442479713505)
        assert np.allclose(size_pix["max_window_scale_0_area"], 1825.7034994476212)
        assert np.allclose(size_pix["min_window_scale_1_center"], 3.499494833498043)
        assert np.allclose(size_pix["min_window_scale_1_fwhm"], 1.7497474167490215)
        assert np.allclose(size_pix["min_window_scale_1_area"], 1.202293800518385)
        assert np.allclose(size_pix["max_window_scale_1_center"], 68.18442479713505)
        assert np.allclose(size_pix["max_window_scale_1_fwhm"], 34.092212398567526)
        assert np.allclose(size_pix["max_window_scale_1_area"], 456.4258748619053)
