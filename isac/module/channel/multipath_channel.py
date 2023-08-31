from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from scipy.constants import c

from ...antenna import Generic3DAntennaArray
from ...antenna.array_utils import format_angle
from ...utils import crandn, exp1j2pi, uniform
from ..ofdm import OFDMConfig


class MultiPathChannelConfig:
    def __init__(
        self,
        path_gains: torch.Tensor | Sequence[complex],
        path_delays: torch.Tensor | Sequence[float],
        doppler_shifts: torch.Tensor | Sequence[float],
        aoas: torch.Tensor | Sequence,
        aods: torch.Tensor | Sequence,
    ) -> None:
        """Multi-path channel configuration, specifed by a number of multi-path parameters, i.e.,
           path gains, path delays, doppler shifts, angle of arrivals and angle of departures. All
           parameters should have the same length, which is the number of path.

        :param path_gains: Path gains
        :type path_gains: torch.Tensor | Sequence[complex]
        :param path_delays: Path delays
        :type path_delays: torch.Tensor | Sequence[float]
        :param doppler_shifts: Doppler shifts
        :type doppler_shifts: torch.Tensor | Sequence[float]
        :param aoas: Angle of arrivals
        :type aoas: torch.Tensor | Sequence
        :param aods: Angle of departures
        :type aods: torch.Tensor | Sequence
        """
        super().__init__()
        assert len(path_gains) == len(path_delays) == len(doppler_shifts) == len(aoas) == len(aods)
        self._num_paths_full = len(path_gains)
        self.path_delays = path_delays  # type: ignore
        self.path_gains = path_gains  # type: ignore
        self.doppler_shifts = doppler_shifts  # type: ignore
        self.aoas = aoas  # type: ignore
        self.aods = aods  # type: ignore

    @property
    def num_paths(self) -> int:
        return len(self.path_gains)

    @property
    def num_eff_paths(self) -> int:
        return len(torch.nonzero(self.path_gains)[0])

    @property
    def path_gains(self) -> torch.Tensor:
        return self._path_gains

    @path_gains.setter
    def path_gains(self, b: torch.Tensor | Sequence[complex]) -> None:
        assert self._num_paths_full == len(b)
        self._path_gains = torch.as_tensor(b)

    @property
    def path_delays(self) -> torch.Tensor:
        return self._path_delays

    @path_delays.setter
    def path_delays(self, tau: torch.Tensor | Sequence[float]) -> None:
        assert self._num_paths_full == len(tau)
        assert torch.isreal(tau).all()
        assert all([x >= 0 for x in tau])
        self._path_delays = torch.as_tensor(tau)

    @property
    def doppler_shifts(self) -> torch.Tensor:
        return self._doppler_shifts

    @doppler_shifts.setter
    def doppler_shifts(self, ds: torch.Tensor | Sequence[float]) -> None:
        assert self._num_paths_full == len(ds)
        assert torch.isreal(ds).all()
        self._doppler_shifts = torch.as_tensor(ds)

    @property
    def aoas(self) -> torch.Tensor:
        return self._aoas

    @aoas.setter
    def aoas(self, theta: torch.Tensor | Sequence) -> None:
        assert self._num_paths_full == len(theta)
        if len(theta) == 0:
            self._aoas = torch.zeros((0, 2))  # otherwise stack will fail
        else:
            theta_list = [format_angle(x) for x in theta]
            self._aoas = torch.stack(theta_list, dim=0)

    @property
    def aods(self) -> torch.Tensor:
        return self._aods

    @aods.setter
    def aods(self, phi: torch.Tensor | Sequence[float]) -> None:
        assert self._num_paths_full == len(phi)
        if len(phi) == 0:
            self._aods = torch.zeros((0, 2))
        else:
            phi_list = [format_angle(x) for x in phi]
            self._aods = torch.stack(phi_list, dim=0)

    def __str__(self) -> str:
        return (
            f"num_paths: {self.num_paths}\n\n"
            f"path_gains: {self.path_gains}\n\n"
            f"path_delays: {self.path_delays}\n\n"
            f"doppler_shifts: {self.doppler_shifts}\n\n"
            f"aoas: {self.aoas}\n\n aods: {self.aods}"
        )

    __repr__ = __str__

    @staticmethod
    def random_generate(
        num_paths: int,
        sampling_time: float = 1 / (1024 * 15e3),
        carrier_frequency: float = 3e9,
    ) -> MultiPathChannelConfig:
        assert isinstance(num_paths, int) and num_paths >= 0
        assert sampling_time > 0
        fc = carrier_frequency
        assert fc > 0
        path_gains = crandn(num_paths)
        aodsaz = uniform(-torch.pi, torch.pi, num_paths)
        aodsel = uniform(-torch.pi / 2, torch.pi / 2, num_paths)
        aods = torch.stack((aodsaz, aodsel), dim=-1)

        aoasaz = uniform(-torch.pi, torch.pi, num_paths)
        aoasel = uniform(-torch.pi / 2, torch.pi / 2, num_paths)
        aoas = torch.stack((aoasaz, aoasel), dim=-1)

        d = uniform(0, 100, num_paths)
        path_delays = d / c
        v = uniform(-80, 80, num_paths)
        doppler_shifts = v * fc / c

        mpc_configs = MultiPathChannelConfig(
            path_gains=path_gains,
            path_delays=path_delays,
            doppler_shifts=doppler_shifts,
            aoas=aoas,
            aods=aods,
        )

        return mpc_configs


class OFDMBeamSpaceChannel(nn.Module):
    def __init__(
        self,
        mpc_configs: MultiPathChannelConfig,
        ofdm_config: OFDMConfig,
        tx_array: Generic3DAntennaArray,
        rx_array: Generic3DAntennaArray,
    ) -> None:
        """Beam space OFDM channel model for 3D Tx/Rx array
        :param mpc_configs: multi-path configurations, should contains path gains, path delays,
                            doppler shifts, aoas and aods
        :type mpc_configs: MultiPathChannelConfig
        :param ofdm_config: OFDM configurations
        :type ofdm_config: OFDMConfig
        :param tx_array: transmit 3D antenna array
        :type tx_array: Generic3DAntennaArray
        :param rx_array: receive 3D antenna array
        :type rx_array: Generic3DAntennaArray
        """
        super().__init__()
        self.mpc_configs = mpc_configs
        self.ofdm_config = ofdm_config
        self.tx_array = tx_array
        self.rx_array = rx_array

    @property
    def tx_array(self) -> Generic3DAntennaArray:
        return self._tx_array

    @tx_array.setter
    def tx_array(self, txa: Generic3DAntennaArray) -> None:
        assert isinstance(txa, Generic3DAntennaArray)
        self._tx_array = txa

    @property
    def rx_array(self) -> Generic3DAntennaArray:
        return self._rx_array

    @rx_array.setter
    def rx_array(self, rxa: Generic3DAntennaArray) -> None:
        assert isinstance(rxa, Generic3DAntennaArray)
        self._rx_array = rxa

    @property
    def mpc_configs(self) -> MultiPathChannelConfig:
        return self._mpc_configs

    @mpc_configs.setter
    def mpc_configs(self, mpcc: MultiPathChannelConfig) -> None:
        assert isinstance(mpcc, MultiPathChannelConfig)
        self._mpc_configs = mpcc

    @property
    def ofdm_config(self) -> OFDMConfig:
        return self._ofdm_config

    @ofdm_config.setter
    def ofdm_config(self, ofdmcon: OFDMConfig) -> None:
        assert isinstance(ofdmcon, OFDMConfig)
        self._ofdm_config = ofdmcon

    def get_channel(
        self,
        num_carriers: int,
        num_symbols: int,
        return_multipath: bool = False,
        dc_in_middle: bool = True,
    ) -> torch.Tensor:
        """return the channel matrix for each OFDM resource elements

        :param num_carriers: number of carriers
        :type num_carriers: int
        :param num_symbols: number of symbols
        :type num_symbols: int
        :param return_multipath: whether to return channel multipath components, defaults to False
        :type return_multipath: bool, optional
        :param dc_in_middle: whether to performing ifftshift on channel matrix to put DC component in the
                            middle of OFDM grid, defaults to True
        :type dc_in_middle: bool, optional

        :return: [num_ant_rx, num_ant_tx, (num_paths), num_carriers, num_symbols], when dc_in_middle is True,
                    the DC component (n=0) is located at the middle of OFDM grid
        :rtype: torch.Tensor
        """
        if self.mpc_configs.num_paths == 0:
            Hf = torch.zeros(
                (self.rx_array.num_antennas, self.tx_array.num_antennas, 1, num_carriers, num_symbols),
            )
        else:
            Atx = self.tx_array.steering_matrix(
                grid=self.mpc_configs.aods,
            )  # num_paths x num_ants_tx
            Arx = self.rx_array.steering_matrix(
                grid=self.mpc_configs.aoas,
            )  # num_paths x num_ants_rx
            A = torch.einsum("...l, ...li, ...lj->...lij", self.mpc_configs.path_gains, Arx, Atx.conj())  # type: ignore

            # assert False
            tt, ff = torch.meshgrid(
                torch.arange(num_symbols) * self.ofdm_config.symbol_time,
                torch.arange(num_carriers) * self.ofdm_config.subcarrier_spacing,
                indexing="xy",
            )
            tt_ = torch.einsum("...nk,...l->...nkl", tt, self.mpc_configs.doppler_shifts)
            ff_ = torch.einsum("...nk,...l->...nkl", ff, self.mpc_configs.path_delays)

            omega_nk = exp1j2pi(tt_ - ff_)

            Hf = torch.einsum(
                "...nkl,...lij->...ijlnk",
                omega_nk,
                A,
            )  # [num_ant_rx x num_ant_tx x num_paths x num_carriers x num_symbols]
        if dc_in_middle:
            Hf = torch.fft.fftshift(Hf, dim=-2)  # make DC component in the middle
        if not return_multipath:
            Hf = torch.einsum("...ijlnk -> ijnk", Hf)
        return Hf

    def forward(self, signal_in: torch.Tensor) -> torch.Tensor:
        """apply channel on input OFDM signal grid

        :param signal_in: [..., Nt, Nfft, Nsym], Nt the number of antennas, Nf the number of subcarriers,
        and T the number of OFDM symbols. DC component in the middle
        :type signal_in: torch.Tensor
        :return: [..., Nr, Nfft, Nsym], Nr the number of receive antennas
        :rtype: torch.Tensor
        """
        num_symbols = signal_in.shape[-1]
        Hf = self.get_channel(num_carriers=self.ofdm_config.Nfft, num_symbols=num_symbols, return_multipath=False)
        out = torch.einsum("...ijnk, ...jnk -> ...ink", Hf, signal_in)

        return out
