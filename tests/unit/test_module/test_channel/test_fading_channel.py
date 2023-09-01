import numpy as np
import pytest
import torch

from isac.module.channel.fading_channel import AWGNChannel, RayleighChannel
from isac.utils import db2lin, lin2db, measure_batch_sig_pow

batch_size = 16


@pytest.mark.parametrize("snr_db", [-10, 0])
@pytest.mark.parametrize("sigpow_db", [10, 0])  # total power over all antennas
def test_AWGNchannel(num_ant_tx, snr_db, sigpow_db, device):
    sig_len = 5000
    sigpow_lin = db2lin(sigpow_db)
    awgn = AWGNChannel(snr_db=snr_db, sigpow_db=sigpow_db)

    signal_in_real = torch.randn(batch_size, num_ant_tx, sig_len) * np.sqrt(sigpow_lin / num_ant_tx)
    signal_in_real = signal_in_real.to(device)

    signal_in_complex = (
        torch.randn(batch_size, num_ant_tx, sig_len) + 1j * torch.randn(batch_size, num_ant_tx, sig_len)
    ) * np.sqrt(
        sigpow_lin / num_ant_tx / 2,
    )
    signal_in_complex = signal_in_complex.to(device)

    noise_out_real = awgn(signal_in_real) - signal_in_real
    noise_power_real = measure_batch_sig_pow(noise_out_real).cpu()
    assert np.isclose(sigpow_db - lin2db(noise_power_real), snr_db, atol=1, rtol=1e-1)

    noise_out_complex = awgn(signal_in_complex) - signal_in_complex
    noise_power_complex = measure_batch_sig_pow(noise_out_complex).cpu()
    assert np.isclose(sigpow_db - lin2db(noise_power_complex), snr_db, atol=1, rtol=1e-1)

    awgn.sigpow_db = "measured"
    noise_out_real = awgn(signal_in_real) - signal_in_real
    noise_power_real = measure_batch_sig_pow(noise_out_real).cpu()
    assert np.isclose(sigpow_db - lin2db(noise_power_real), snr_db, atol=1, rtol=1e-1)

    noise_out_complex = awgn(signal_in_complex) - signal_in_complex
    noise_power_complex = measure_batch_sig_pow(noise_out_complex).cpu()
    assert np.isclose(sigpow_db - lin2db(noise_power_complex), snr_db, atol=1, rtol=1e-1)


@pytest.mark.parametrize("sigpow_db", [-10, 0])
@pytest.mark.parametrize("pl_db", [10])
def test_rayleigh(num_ant_tx, num_ant_rx, sigpow_db, pl_db, device):
    sig_len = 10
    sigpow_lin = db2lin(sigpow_db)
    num_trials = 500

    rl_chan = RayleighChannel(path_loss=pl_db, num_ant_rx=num_ant_rx)

    sig_outpow_complex = 0
    sig_outpow_real = 0

    for _ in range(num_trials):
        signal_in_complex = (
            torch.randn(batch_size, num_ant_tx, sig_len) + 1j * torch.randn(batch_size, num_ant_tx, sig_len)
        ) * np.sqrt(
            sigpow_lin / num_ant_tx / 2,
        )

        signal_in_complex = signal_in_complex.to(device)
        signal_out_complex = rl_chan(signal_in_complex)
        sig_outpow_complex += measure_batch_sig_pow(signal_out_complex)
        assert signal_out_complex.shape[-2:] == (num_ant_rx, sig_len)

        signal_in_real = torch.randn(batch_size, num_ant_tx, sig_len) * np.sqrt(sigpow_lin / num_ant_tx)
        signal_in_real = signal_in_real.to(device)
        signal_out_real = rl_chan(signal_in_real)
        sig_outpow_real += measure_batch_sig_pow(signal_out_real)
        assert signal_out_real.shape[-2:] == (num_ant_rx, sig_len)

    sig_outpow_complex_db = lin2db(sig_outpow_complex / num_trials).cpu()
    assert np.isclose(sigpow_db - sig_outpow_complex_db, pl_db, atol=1, rtol=2e-1)

    sig_outpow_real_db = lin2db(sig_outpow_real / num_trials).cpu()
    assert np.isclose(sigpow_db - sig_outpow_real_db, pl_db, atol=1, rtol=2e-1)
