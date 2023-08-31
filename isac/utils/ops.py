from __future__ import annotations

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
