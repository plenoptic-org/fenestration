import pytest
import torch

import fenestration as fen

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="package")
def rand_img():
    return torch.rand((1, 1, 256, 256), dtype=torch.float32)


@pytest.fixture(scope="package")
def pool_win(rand_img):
    return fen.PoolingWindows(0.5, rand_img.shape[2:], num_scales=2)
