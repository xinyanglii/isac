import random

import pytest
import torch

from isac.estimator.functional import estimate_subspace_order, music_spectrum
from isac.mimo.antenna import UniformLinearArray


@pytest.fixture
def x(device):
    batch_size = 16
    num_ant_tx = 8
    num_samples = 1000
    return torch.randn(batch_size, num_ant_tx, num_samples, dtype=torch.complex64, device=device)


@pytest.mark.parametrize("method", ["mdl", "aic"])
def test_estimate_subspace_order_and_music_spectrum(x, method):
    batch_size, num_ant_tx = x.shape[:2]
    num_ant_rx = num_ant_tx - 2
    num_paths = [random.randint(1, num_ant_rx - 2) for _ in range(batch_size)]
    rx_array = UniformLinearArray(num_ant_rx)
    y = torch.zeros(batch_size, num_ant_rx, *x.shape[2:], dtype=torch.complex64, device=x.device)
    aoas_list = []
    for i, num_path in enumerate(num_paths):
        aoas = torch.linspace(-torch.pi / 2 + 1, torch.pi / 2 - 5e-1, num_path)
        aoas_list.append(aoas)
        rx_stm = rx_array.steering_matrix(aoas, axis=1).to(x.device)
        randmat = torch.randn(num_path, num_ant_tx, dtype=torch.complex64, device=x.device)
        y[i] = rx_stm @ randmat @ x[i]

    y_noisy = y + torch.randn_like(y)
    num_signal_source = estimate_subspace_order(y_noisy, method=method)
    assert num_signal_source.shape == (batch_size,)

    # We give a tolerance of 1, because the estimation is not always accurate
    assert all([num_signal_source[i] in [num_paths[i], num_paths[i] - 1, num_paths[i] + 1] for i in range(batch_size)])

    scan_grid = torch.linspace(-torch.pi / 2 + 1e-3, torch.pi / 2 - 1e-3, 100)
    rx_stm = rx_array.steering_matrix(scan_grid, axis=1).to(x.device)
    spec = music_spectrum(y_noisy, num_paths, rx_stm)
    assert spec.shape == (batch_size, len(scan_grid))
    peaks = torch.argmax(spec, dim=1)
    for i in range(batch_size):
        assert torch.isclose(scan_grid[peaks[i]], aoas_list[i], rtol=1e-1).any()
