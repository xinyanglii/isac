from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OFDMConfig:
    subcarrier_spacing: float = 15e3
    cp_frac: float = 0.07
    num_guard_carriers: tuple[int, int] = (0, 0)
    Nfft: int = 64

    def __post_init__(self):
        assert self.subcarrier_spacing > 0
        assert 0 <= self.cp_frac <= 1
        assert len(self.num_guard_carriers) == 2 and all(isinstance(x, int) and x >= 0 for x in self.num_guard_carriers)
        assert isinstance(self.Nfft, int) and self.Nfft > sum(self.num_guard_carriers)

    @property
    def cp_len(self):
        return int(np.ceil(self.Nfft * self.cp_frac))

    @property
    def num_samples_per_sym(self):
        return self.Nfft + self.cp_len

    @property
    def num_data_carriers(self):
        return self.Nfft - sum(self.num_guard_carriers)

    @property
    def symbol_time(self):
        return 1 / self.subcarrier_spacing

    @property
    def sampling_time(self):
        return self.symbol_time / self.Nfft

    # TODO: This function can be extended to a class in the future
    def get_resource_grid(self, waveform: torch.Tensor):
        num_carriers = waveform.shape[-2]
        assert num_carriers == self.num_data_carriers
        grid = F.pad(waveform, (0, 0, self.num_guard_carriers[0], self.num_guard_carriers[1]), "constant", 0)
        return grid

    def get_data_grid(self, waveform: torch.Tensor):
        num_carriers = waveform.shape[-2]
        assert num_carriers == self.Nfft
        pad_len = self.num_guard_carriers
        if pad_len[1] != 0:
            grid = waveform[..., pad_len[0] : -pad_len[1], :]
        else:
            grid = waveform[..., pad_len[0] :, :]
        return grid


# TODO: variable CP length
class OFDMModulator(nn.Module):
    def __init__(
        self,
        config: OFDMConfig = OFDMConfig(),
    ) -> None:
        """OFDM modulator class

        :param config: OFDM configuration object, in which scs and cp_len is defined.
        :type config: OFDMConfig, optional, defaults to OFDMConfig()
        """
        super().__init__()
        self.config = config

    # TODO: add windowing function
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """modulate waveform, i.e., OFDM resource grid into time domain

        :param waveform: OFDM resource grid, of size [..., num_carriers, num_symbols], the DC component in middle
        :type waveform: torch.Tensor
        :return: modulated OFDM waveform in time doamin, of shape [..., num_samples]
        :rtype: torch.Tensor
        """
        out = ofdmmodulate(waveform, self.config)
        return out


class OFDMDemodulator(nn.Module):
    def __init__(
        self,
        config: OFDMConfig = OFDMConfig(),
    ) -> None:
        """OFDM demodulator class

        :param config: OFDM configuration object
        :type config: OFDMConfig, optional, defaults to OFDMConfig()
        """
        super().__init__()
        self.config = config

    # TODO: add windowing function
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """demodulate time domain OFDM waveform into frequency domain, waveform -> waveform_f

        :param waveform: OFDM waveform in time domain, of shape [..., num_samples]
        :type waveform: torch.Tensor
        :return: OFDM waveform in frequency domain, i.e., OFDM resource grid, of shape
                    [..., num_carriers, num_symbols], DC component is in the middle of subcarriers
        :rtype: torch.Tensor
        """
        num_wf_symbols = waveform.shape[-1] // self.config.num_samples_per_sym
        out = ofdmdemodulate(waveform[..., : num_wf_symbols * self.config.num_samples_per_sym], self.config)

        return out


def ofdmmodulate(waveform: torch.Tensor, ofdm_config: OFDMConfig) -> torch.Tensor:
    # pad_len = ofdm_config.num_guard_carriers
    # grid = np.pad(waveform, ((0, 0), (pad_len[0], pad_len[1]), (0, 0)), "constant", constant_values=0)
    grid = ofdm_config.get_resource_grid(waveform)

    grid_to_ifft = torch.fft.ifftshift(grid, dim=-2)
    gridifft = torch.fft.ifft(grid_to_ifft, n=ofdm_config.Nfft, dim=-2) * (ofdm_config.Nfft**0.5)
    cp = gridifft[..., -ofdm_config.cp_len :, :]
    grid_w_cp = torch.cat([cp, gridifft], dim=-2)

    out = grid_w_cp.transpose(-1, -2).flatten(-2)
    return out


def ofdmdemodulate(waveform: torch.Tensor, ofdm_config: OFDMConfig) -> torch.Tensor:
    wf_wo_cp = waveform.reshape((*waveform.shape[:-1], -1, ofdm_config.Nfft + ofdm_config.cp_len)).transpose(-1, -2)[
        ...,
        ofdm_config.cp_len :,
        :,
    ]
    wf_f = torch.fft.fftshift(torch.fft.fft(wf_wo_cp, n=ofdm_config.Nfft, dim=-2), dim=-2) / np.sqrt(ofdm_config.Nfft)
    # grid = wf_f
    # pad_len = ofdm_config.num_guard_carriers
    # if pad_len[1] != 0:
    #     grid = wf_f[:, pad_len[0] : -pad_len[1], :]
    # else:
    #     grid = wf_f[:, pad_len[0] :, :]
    # assert False
    grid = ofdm_config.get_data_grid(wf_f)

    return grid
