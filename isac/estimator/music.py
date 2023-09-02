from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from ..mimo import UniformLinearArray
from ..utils import find_peaks
from .functional import music_spectrum


class MUSICAoAEstimator1D(nn.Module):
    def __init__(
        self,
        antenna_array: UniformLinearArray,
        scan_grid: Sequence[float] | torch.Tensor,
    ) -> None:
        """Estimate the angle of arrival using 1D MUSIC algorithm assuming a uniform linear array.

        :param antenna_array: Uniform linear array object
        :type antenna_array: :class:`isac.mimo.UniformLinearArray`
        :param scan_grid: angles to compute the MUSIC spectrum, in radians
        :type scan_grid: Sequence[float] | torch.Tensor
        """
        super().__init__()
        self.antenna_array = antenna_array
        self.scan_grid = scan_grid

    @property
    def antenna_array(self) -> UniformLinearArray:
        return self._antenna_array

    @antenna_array.setter
    def antenna_array(self, antarr: UniformLinearArray) -> None:
        assert isinstance(antarr, UniformLinearArray)
        self._antenna_array = antarr

    @property
    def scan_grid(self) -> torch.Tensor:
        return self._scan_grid

    @scan_grid.setter
    def scan_grid(self, sg: torch.Tensor) -> None:
        assert sg.ndim == 1
        assert torch.max(sg) <= torch.pi / 2 and torch.min(sg) >= -torch.pi / 2
        self._scan_grid = torch.asarray(sg)

    @property
    def steering_mat(self) -> torch.Tensor:
        return self.antenna_array.steering_matrix(self.scan_grid, axis=1)

    def forward(
        self,
        x: torch.Tensor,
        num_signal_source: Sequence[int] | torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param x: input signal, shape [batch_size, num_ant, ...]
        :type x: torch.Tensor
        :param num_signal_source: number of signal sources, shape [batch_size], can be obtained by
            :func:`estimate_subspace_order`
        :type num_signal_source: Sequence[int] | torch.Tensor
        :return: estimated angles of arrival for each signal in the batch, each of shape [num_signal_source],
            and MUSIC spectrum, shape [batch_size, len_scan_grid]
        :rtype: Tuple[List[torch.Tensor], torch.Tensor]
        """
        spectrum = music_spectrum(x, num_signal_source, self.steering_mat)  # [B, len_scan_grid]
        angs = []
        for i, spec in enumerate(spectrum):
            peaks, pidx = find_peaks(spec)
            sortidx = torch.argsort(peaks, descending=True)
            n = min(num_signal_source[i], len(sortidx))
            angs.append(self.scan_grid[pidx[sortidx[:n]]])
        return angs, spectrum
