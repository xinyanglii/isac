import numpy as np
import pytest
import torch

from isac.channel import MultiPathChannelConfig, OFDMBeamSpaceChannel, generate_multipath_ofdm_channel
from isac.mimo.antenna import UniformCubicArray, UniformLinearArray, UniformRectangularArray
from isac.ofdm.ofdm import OFDMConfig
from isac.utils import crandn, uniform


def test_mpc_config(num_paths):
    rconfig = MultiPathChannelConfig.random_generate(num_paths=num_paths)
    assert (
        len(rconfig.path_gains)
        == len(rconfig.path_delays)
        == len(rconfig.doppler_shifts)
        == len(rconfig.aoas)
        == len(rconfig.aods)
        == num_paths
    )
    assert rconfig.aoas.shape == rconfig.aods.shape == (num_paths, 2)

    b = crandn(num_paths)
    tau = uniform(0, 1e-7, num_paths)
    fD = uniform(-1000, 1000, num_paths)
    aoas = uniform(-torch.pi / 2, torch.pi / 2, num_paths)
    aods = uniform(-torch.pi / 2, torch.pi / 2, num_paths)

    config = MultiPathChannelConfig(b, tau, fD, aoas, aods)

    assert config.aoas.shape == config.aods.shape == (num_paths, 2)

    assert isinstance(config.path_gains, torch.Tensor)
    assert isinstance(config.path_delays, torch.Tensor)
    assert isinstance(config.doppler_shifts, torch.Tensor)
    assert isinstance(config.aoas, torch.Tensor)
    assert isinstance(config.aods, torch.Tensor)


@pytest.mark.parametrize("TXArray", [UniformLinearArray, UniformRectangularArray, UniformCubicArray])
@pytest.mark.parametrize("RXArray", [UniformLinearArray, UniformRectangularArray, UniformCubicArray])
def test_beam_space_channel(
    num_paths,
    num_ant_tx,
    num_ant_rx,
    scs,
    num_carriers,
    num_symbols,
    TXArray,
    RXArray,
    device,
):
    batch_size = 16
    sampling_time = 1 / scs / num_carriers
    symbol_duration = 1 / scs

    ofdmconf = OFDMConfig(subcarrier_spacing=scs, num_guard_carriers=(0, 0), Nfft=num_carriers)

    if TXArray is UniformLinearArray:
        tx_array = TXArray(num_antennas=num_ant_tx)
    elif TXArray is UniformRectangularArray:
        tx_array = TXArray(antenna_dimension=(num_ant_tx, np.random.randint(1, 8)))
    elif TXArray is UniformCubicArray:
        tx_array = TXArray(antenna_dimension=(num_ant_tx, *np.random.randint(1, 8, size=2)))

    num_ant_tx = tx_array.num_antennas

    if RXArray is UniformLinearArray:
        rx_array = RXArray(num_antennas=num_ant_rx)
    elif RXArray is UniformRectangularArray:
        rx_array = RXArray(antenna_dimension=(num_ant_rx, np.random.randint(1, 8)))
    elif RXArray is UniformCubicArray:
        rx_array = RXArray(antenna_dimension=(num_ant_rx, *np.random.randint(1, 8, size=2)))

    num_ant_rx = rx_array.num_antennas

    mpc_configs = MultiPathChannelConfig.random_generate(
        num_paths=num_paths,
        sampling_time=sampling_time,
    )
    obschannel = OFDMBeamSpaceChannel(
        mpc_configs=mpc_configs,
        ofdm_config=ofdmconf,
        tx_array=tx_array,
        rx_array=rx_array,
    )
    H = generate_multipath_ofdm_channel(
        tx_array=tx_array,
        rx_array=rx_array,
        num_carriers=num_carriers,
        num_symbols=num_symbols,
        mpc_configs=mpc_configs,
        symbol_time=symbol_duration,
        subcarrier_spacing=scs,
    )

    H_wo_shift = torch.fft.ifftshift(H, dim=-2)

    At = [tx_array.steering_vector(aod) for aod in mpc_configs.aods]
    Ar = [rx_array.steering_vector(aoa) for aoa in mpc_configs.aoas]
    path_gains = mpc_configs.path_gains
    doppler_shifts = mpc_configs.doppler_shifts
    path_delays = mpc_configs.path_delays

    signal_in = crandn((batch_size, num_ant_tx, num_carriers, num_symbols), device=device)
    signal_in_wo_shift = torch.fft.ifftshift(signal_in, dim=-2)
    signal_out = obschannel(signal_in)
    signal_out_wo_shift = torch.fft.ifftshift(signal_out, dim=-2)

    for n in range(num_carriers):
        for k in range(num_symbols):
            H_true = torch.zeros((num_ant_rx, num_ant_tx), dtype=torch.complex64)
            for path in range(num_paths):
                H_true += (
                    path_gains[path]
                    * torch.exp(
                        1j * 2 * torch.pi * (doppler_shifts[path] * symbol_duration * k - path_delays[path] * scs * n),
                    )
                    * torch.outer(Ar[path], At[path].conj())
                )
            assert torch.allclose(H_wo_shift[:, :, n, k], H_true, rtol=1e-1)
            signal_out_true = torch.einsum("...i,bi...->b...", H_true.to(signal_in), signal_in_wo_shift[..., n, k])
            assert torch.allclose(signal_out_wo_shift[..., n, k], signal_out_true, rtol=1e-1)
