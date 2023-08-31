from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch
from scipy.constants import c


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


@pytest.fixture
def allcomb(request):
    return request.config.getoption("--all")


carrier_frequency_list = [2.4e9, 1e3, c]
Nfft_list = [64, 31]
num_carriers_list = [1, 16]
num_symbols_list = [1, 16]
num_ant_tx_list = [1, 8]
num_ant_rx_list = [1, 8]
scs_list = [1.0, 15e3]
cp_frac_list = np.linspace(1e-2, 1, 5)
num_guard_carriers_list = list(product([0, 1, 8], [0, 1, 8]))
num_paths_list = [3, 1, 0, 8]
dc_null_list = [True, False]

device_list = [torch.device("cpu")] + [torch.device(i) for i in range(torch.cuda.device_count())]


def get_params(request, allcomb, default):
    if allcomb:
        return request.param
    else:
        if request.param != default:
            pytest.skip("All combinations will be run with --all option")
        return default


@pytest.fixture(params=carrier_frequency_list)
def carrier_frequency(request, allcomb):
    return get_params(request, allcomb, carrier_frequency_list[0])


@pytest.fixture(params=Nfft_list)
def Nfft(request, allcomb):
    return get_params(request, allcomb, Nfft_list[0])


@pytest.fixture(params=num_carriers_list)
def num_carriers(request, allcomb):
    return get_params(request, allcomb, num_carriers_list[0])


@pytest.fixture(params=num_symbols_list)
def num_symbols(request, allcomb):
    return get_params(request, allcomb, num_symbols_list[0])


@pytest.fixture(params=num_ant_tx_list)
def num_ant_tx(request, allcomb):
    return get_params(request, allcomb, num_ant_tx_list[0])


@pytest.fixture(params=num_ant_rx_list)
def num_ant_rx(request, allcomb):
    return get_params(request, allcomb, num_ant_rx_list[0])


@pytest.fixture(params=scs_list)
def scs(request, allcomb):
    return get_params(request, allcomb, scs_list[0])


@pytest.fixture(params=cp_frac_list)
def cp_frac(request, allcomb):
    return get_params(request, allcomb, cp_frac_list[0])


@pytest.fixture(params=num_guard_carriers_list)
def num_guard_carriers(request, allcomb):
    return get_params(request, allcomb, num_guard_carriers_list[0])


@pytest.fixture(params=num_paths_list)
def num_paths(request, allcomb):
    return get_params(request, allcomb, num_paths_list[0])


@pytest.fixture(params=device_list)
def device(request, allcomb):
    return get_params(request, allcomb, device_list[0])
