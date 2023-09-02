from __future__ import annotations

from typing import Tuple

import torch
from scipy.constants import c


def db2lin(x):
    return 10 ** (x / 10)


def lin2db(x):
    return 10 * torch.log10(x)


def measure_batch_sig_pow(x):
    """Compute the empirical power of a tensor, assume the tensor is of size
       [B,...,T], where B is the batch size, T is the number of time steps.

    :param x: input tensor, of size [B,...,T]
    :type x: torch.Tensor
    """
    return x.norm() ** 2 / (x.shape[0] * x.shape[-1])


def exp1j2pi(x):
    return torch.exp(1j * 2 * torch.pi * x)


def freq2wavelen(fc):
    return c / fc


def wavelength2freq(wavelength):
    return c / wavelength


def uniform(low=0.0, high=1.0, size=1, dtype=torch.float32, device=None):
    return torch.rand(size, dtype=dtype, device=device) * (high - low) + low


def crandn(size=1, device=None):
    return torch.randn(size, device=device) + 1j * torch.randn(size, device=device)


def kron3(a, b, c):
    return torch.kron(torch.kron(a, b), c)


# TODO: add more features such as detection threshold, minimum distance between peaks, etc.
def find_peaks(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find local maxima in a 1D tensor.
    See https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404
    and
    https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_peak_finding_utils.pyx#L20

    :param x: input iD tensor
    :type x: torch.Tensor
    :return: peak values and indices
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    peak_idx = torch.zeros_like(x, dtype=torch.bool)
    peak_idx[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    return x[peak_idx], torch.nonzero(peak_idx, as_tuple=True)[0]
