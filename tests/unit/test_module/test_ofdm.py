import math

import numpy as np
import pytest
import torch

from isac.module.ofdm import OFDMConfig, OFDMDemodulator, OFDMModulator
from isac.utils import crandn


@pytest.mark.parametrize("Nfft", [31, 64] + [1, 2])
def test_ofdm_config(Nfft, scs, cp_frac, num_guard_carriers, num_ant_tx, num_symbols, device):
    if Nfft <= sum(num_guard_carriers):
        with pytest.raises(Exception):
            ofdmconf = OFDMConfig(
                subcarrier_spacing=scs,
                cp_frac=cp_frac,
                num_guard_carriers=num_guard_carriers,
                Nfft=Nfft,
            )
    else:
        ofdmconf = OFDMConfig(subcarrier_spacing=scs, cp_frac=cp_frac, num_guard_carriers=num_guard_carriers, Nfft=Nfft)
        assert ofdmconf is not None

        num_data_carriers = Nfft - sum(num_guard_carriers)
        assert num_data_carriers == ofdmconf.num_data_carriers
        wf = torch.randn((num_ant_tx, num_data_carriers, num_symbols), device=device)
        resource_grid = ofdmconf.get_resource_grid(wf)
        data_grid = ofdmconf.get_data_grid(resource_grid)
        assert torch.allclose(wf, data_grid)

        wf = torch.randn((16, num_ant_tx, num_data_carriers, num_symbols), device=device)
        resource_grid = ofdmconf.get_resource_grid(wf)
        data_grid = ofdmconf.get_data_grid(resource_grid)
        assert torch.allclose(wf, data_grid)


def test_ofdm_modem(Nfft, num_symbols, num_ant_tx, scs, cp_frac, num_guard_carriers, device):
    num_ant = num_ant_tx
    ofdmconf = OFDMConfig(subcarrier_spacing=scs, cp_frac=cp_frac, num_guard_carriers=num_guard_carriers, Nfft=Nfft)
    num_carriers = ofdmconf.num_data_carriers
    num_samples_per_sym = ofdmconf.num_samples_per_sym
    cp_len = ofdmconf.cp_len

    ofdmmod = OFDMModulator(config=ofdmconf)
    ofdmdem = OFDMDemodulator(config=ofdmconf)

    wf_f = crandn((16, num_ant, num_carriers, num_symbols), device=device)

    wf_t = ofdmmod(wf_f)
    assert wf_t.shape[-2:] == (num_ant, num_symbols * num_samples_per_sym)

    for k in range(num_symbols):
        assert torch.allclose(
            wf_t[..., k * num_samples_per_sym : k * num_samples_per_sym + cp_len],
            wf_t[..., (k + 1) * num_samples_per_sym - cp_len : (k + 1) * num_samples_per_sym],
        )

    wf_f_dem = ofdmdem(wf_t)
    assert wf_f_dem.shape == wf_f.shape
    assert torch.allclose(wf_f_dem, wf_f, rtol=2e-3)

    if num_guard_carriers[0] == num_guard_carriers[1]:
        deltasig = torch.zeros((num_ant, num_carriers, num_symbols), dtype=torch.complex64, device=device)
        deltasig[:, int(math.ceil((num_carriers - 1) / 2)), :] = Nfft
        deltasig_t = ofdmmod(waveform=deltasig)
        assert np.allclose(deltasig_t.cpu(), Nfft**0.5)

        deltasig_f_dem = ofdmdem(deltasig_t)
        assert torch.allclose(deltasig_f_dem, deltasig, atol=1e-5)
