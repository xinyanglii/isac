from __future__ import annotations

import warnings
from typing import Union

import torch


def format_angle(angle: Union[float, torch.Tensor]) -> torch.Tensor:
    """Format and return the input angle. If angle is scalar, it is considered as elevation angle by default.
        The output angle is a 1D array of length 2, [azimuth, elevation]. The azimuth angle is defined as
        a angle with positive direction from positive x axis to positive y axis, in range [-pi, pi], and the
        elevation angle positive direction is from positive x axis to positive z axis, in range [-pi/2, pi/2].

    :param angle: length 2 1D tensor, sequence of two floats, or a float.
    :type angle: Union[float, torch.Tensor]
    :return: length 2 1D tensor, denoting two angles [azimuth, elevation]
    :rtype: torch.Tensor
    """

    angle = torch.as_tensor(angle)

    angle_formatted = torch.zeros(2).to(angle)  # [azimuth, elevation]
    if angle.numel() == 1:
        angle_formatted[1] = angle
    else:
        angle_formatted = angle

    assert angle_formatted.shape == (2,)
    assert torch.isreal(angle_formatted).all()

    if not (-torch.pi <= angle_formatted[0] <= torch.pi or -torch.pi <= angle_formatted[1] <= torch.pi):
        warnings.warn(UserWarning("The input angle is not in the valid range, will be wrapped"))

    # wrap the elevation angle to [-pi, pi]
    angle_formatted[1] = angle_formatted[1] % (2 * torch.pi)
    if angle_formatted[1] > torch.pi:
        angle_formatted[1] = angle_formatted[1] - 2 * torch.pi
    # if elevation angle is out of range [-pi/2, pi/2], then wrap it and change azimuth angle by pi
    if angle_formatted[1] > torch.pi / 2:
        warnings.warn(
            UserWarning("The elevation angle is not in [-pi/2, pi/2], will be wrapped and add pi to azimuth angle"),
        )
        angle_formatted[1] = torch.pi - angle_formatted[1]
        angle_formatted[0] = angle_formatted[0] + torch.pi
    elif angle_formatted[1] < -torch.pi / 2:
        warnings.warn(
            UserWarning("The elevation angle is not in [-pi/2, pi/2], will be wrapped and add pi to azimuth angle"),
        )
        angle_formatted[1] = -torch.pi - angle_formatted[1]
        angle_formatted[0] = angle_formatted[0] + torch.pi

    # wrap the azimuth angle to [-pi, pi]
    angle_formatted[0] = angle_formatted[0] % (2 * torch.pi)
    if angle_formatted[0] > torch.pi:
        angle_formatted[0] = angle_formatted[0] - 2 * torch.pi

    return angle_formatted
