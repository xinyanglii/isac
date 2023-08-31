from __future__ import annotations

from typing import Union

import torch

from ..utils import exp1j2pi
from .array_utils import format_angle
from .base import AbstractAntennaArray


class Generic3DAntennaArray(AbstractAntennaArray):
    def __init__(
        self,
        element_locations: torch.Tensor,
        wavelength: float = 1,
    ) -> None:
        super().__init__(element_locations=element_locations, wavelength=wavelength)

    def steering_vector(
        self,
        angle: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Compute steering vector given a angle of arrive and wavelength. Reference to
           Optimum Array Processing, Page 29-30, or Matlab function steervec

        :param angle: length 2 1D array or list of two float numbers, or a scalar.
        :type angle: Union[float, torch.Tensor]
        :return: steering vector, 1D array of length number of antennas
        :rtype: torch.Tensor
        """
        angle = format_angle(angle)
        return stv(element_locations=self.element_locations, angle=angle, wavelength=self.wavelength)

    def steering_matrix(
        self,
        grid: torch.Tensor,
        axis: int = 0,
    ) -> torch.Tensor:
        """Generate steering matrix given angle grid and stacking axis

        :param grid: Grid of angles on which steering vectors are computed, grid[i] is the i-the angle,
        :type grid: torch.Tensor
        :param axis: axis along which steering vectors are stacked, defaults to 0
        :type axis: int, optional
        :return: stacked steering matrix
        :rtype: torch.Tensor
        """
        steering_list = [self.steering_vector(a) for a in grid]
        steering_matrix = torch.stack(steering_list, dim=axis)
        return steering_matrix


def stv(
    element_locations: torch.Tensor,
    angle: torch.Tensor,
    wavelength: float = 1.0,
) -> torch.Tensor:
    """Compute steering vector given a angle of arrive and wavelength. Reference to
    Optimum Array Processing, Page 29-30, or Matlab function steervec

    :param element_locations: Tensor with shape (N, 3) where N is the number of antenna elements
    :type element_locations: torch.Tensor
    :param angle: Tensor with shape (2,) corresponding to the azimuth and elevation angles
    :type angle: torch.Tensor
    :param wavelength: Wavelength, defaults to 1
    :type wavelength: float, optional
    :return: Array steering vector in the azimuth and elevation direction given as a tensor of shape (N,)
    :rtype: torch.Tensor
    """
    az = angle[0]
    el = angle[1]
    dirvec = torch.zeros(3).to(angle)
    cos_el, sin_el = torch.cos(el), torch.sin(el)
    cos_az, sin_az = torch.cos(az), torch.sin(az)
    dirvec[0], dirvec[1], dirvec[2] = -cos_el * cos_az, -cos_el * sin_az, -sin_el
    tau = element_locations @ dirvec / wavelength
    sv = exp1j2pi(-tau)
    return sv
