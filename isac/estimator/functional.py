from __future__ import annotations

import math
from typing import Literal, Sequence

import torch


def estimate_subspace_order(
    x: torch.Tensor,
    method: Literal["mdl", "aic"] = "mdl",
) -> torch.Tensor:
    """Estimate the dimension of null space of batched input signals,
       the second dimension is assumed to be the spatial dimension.
       See: https://www.mathworks.com/help/phased/ref/mdltest.html

    :param x: input signal, shape [batch_size, num_ant, ...]
    :type x: torch.Tensor
    :return: estimated dimensions of null space, shape [batch_size], each element is an integer
    :rtype: torch.Tensor
    """
    signal_in = x.flatten(start_dim=2)
    (B, N, K) = signal_in.shape
    cov = signal_in @ signal_in.transpose(1, 2).conj() / K
    eigvs = torch.linalg.eigvalsh(cov).flip(-1)  # [B, N]

    cost = torch.zeros(B, N, device=x.device)
    for d in range(N):
        num = torch.mean(eigvs[:, d:], dim=1)
        den = torch.prod(eigvs[:, d:], dim=1) ** (1 / (N - d))
        cost[:, d] = (N - d) * K * torch.log(num / den)

        if method == "mdl":
            cost[:, d] = cost[:, d] + 1 / 2 * (d * (2 * N - d) + 1) * math.log(K)
        elif method == "aic":
            cost[:, d] = cost[:, d] + d * (2 * N - d)
        else:
            raise ValueError("Unsupported method: {}".format(method))
    # assert False
    return torch.argmin(cost, dim=1)


def music_spectrum(
    signal_in: torch.Tensor,
    num_signal_source: Sequence[int] | torch.Tensor,
    steering_mat: torch.Tensor,
) -> torch.Tensor:
    """Compute the MUSIC spectrum for a given signal and steering matrix. The steering matrix is of size
        [num_ant, len_scan_grid] and the signal is of size [batch_size, num_ant, ...].

    :param signal_in: input signal, shape [batch_size, num_ant, ...]
    :type signal_in: torch.Tensor
    :param num_signal_source: number of subspace order, of shape [batch_size], can be obtained by
            :func:`estimate_subspace_order`
    :type num_signal_source: Sequence[int] | torch.Tensor
    :param steering_mat: steering matrix, shape [num_ant, len_scan_grid]
    :type steering_mat: torch.Tensor
    :return: MUSIC spectrum, shape [batch_size, len_scan_grid]
    :rtype: torch.Tensor
    """
    signal_in = signal_in.flatten(start_dim=2)
    (B, N, K) = signal_in.shape
    signal_cov = signal_in @ signal_in.mH / K
    U = torch.linalg.eigh(signal_cov)[1].flip(-1)  # [B, N, N]
    for i, Ui in enumerate(U):
        Ui[..., : num_signal_source[i]] = 0
    D = torch.linalg.norm(U.mH @ steering_mat, dim=1) ** 2 + torch.finfo(float).eps  # [B, len_scan_grid]
    spec = 1 / D
    return spec
