from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import crandn, db2lin, measure_batch_sig_pow


class AWGNChannel(nn.Module):
    """
    Addtive white Gaussian noise channel class
    """

    def __init__(self, snr_db: float, sigpow_db: float | str = "measured") -> None:
        """
        :param snr_db: SNR in dB
        :type snr_db: float
        :param sigpow_db: signal power in dB, "measured" means measured from the input signal
        :type sigpow_db: float | str
        """
        super().__init__()
        self.snr_db = snr_db
        self.sigpow_db = sigpow_db

    @property
    def sigpow_db(self) -> float | str:
        return self._sigpow_db

    @sigpow_db.setter
    def sigpow_db(self, sp_db: float | str) -> None:
        if isinstance(sp_db, str):
            assert sp_db == "measured"
        else:
            assert isinstance(sp_db, (int, float))
        self._sigpow_db = sp_db

    @property
    def snr_lin(self) -> float:
        return db2lin(self.snr_db)

    def forward(self, signal_in: torch.Tensor) -> torch.Tensor:
        """Add noise to the input signals

        :param signal_in: input signals, of size [B, Nt, ..., T] where B is the batch size,
                            Nt is the number of antenna, T is the number of time steps
        :type signal_in: torch.Tensor
        :return: signals after adding noise
        :rtype: torch.Tensor
        """

        if self.sigpow_db == "measured":
            sp_lin = measure_batch_sig_pow(signal_in)
        else:
            sp_lin = db2lin(self.sigpow_db)

        noisepow = sp_lin / self.snr_lin
        if torch.isreal(signal_in).all():
            noise = (noisepow / signal_in[0, ..., 0].numel()) ** (1 / 2) * torch.randn(
                signal_in.shape,
                device=signal_in.device,
            )
        else:
            noise = (noisepow / signal_in[0, ..., 0].numel() / 2) ** (1 / 2) * crandn(
                signal_in.shape,
                device=signal_in.device,
            )

        return signal_in + noise


class RayleighChannel(nn.Module):
    """
    Rayleigh MIMO channel
    """

    def __init__(
        self,
        path_loss: float,
        num_ant_rx: int = 2,
    ) -> None:
        """
        :param path_loss: path loss in dB
        :type path_loss: float
        :param num_ant_rx: number of receive antennas, defaults to 2
        :type num_ant_rx: int, optional
        """
        super().__init__()
        self.path_loss = path_loss
        self.num_ant_rx = num_ant_rx

    @property
    def path_loss_lin(self) -> float:
        return db2lin(self.path_loss)

    # TODO: let updated H correlated with previous H
    def getH(self, num_ant_tx, is_sig_real: bool, device) -> torch.Tensor:
        if is_sig_real:
            channel_matrix = (1 / self.path_loss_lin / self.num_ant_rx) ** (1 / 2) * torch.randn(
                (self.num_ant_rx, num_ant_tx),
                device=device,
            )
        else:
            channel_matrix = (1 / self.path_loss_lin / self.num_ant_rx / 2) ** (1 / 2) * crandn(
                (self.num_ant_rx, num_ant_tx),
                device=device,
            )
        return channel_matrix

    def forward(self, signal_in: torch.Tensor) -> torch.Tensor:
        """Apply Rayleigh fading channel to the input signal, signal of size [B, Nt, ...]

        :param signal_in: input signal, of size [B, Nt, ...]
        :type signal_in: torch.Tensor
        :return: output signal, of size [B, Nr, ...], where Nr is the number of receive antennas
        :rtype: torch.Tensor
        """
        num_ant_tx = signal_in.shape[1]
        is_sig_real = torch.isreal(signal_in).all()
        channel_mat = self.getH(num_ant_tx, is_sig_real, signal_in.device)
        signal_out = torch.einsum("ij,bj...->bi...", channel_mat, signal_in)
        return signal_out
