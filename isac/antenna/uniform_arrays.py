from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

import torch

from ..utils import exp1j2pi, kron3
from .array_utils import format_angle
from .generic_3d_array import Generic3DAntennaArray


class UniformCubicArray(Generic3DAntennaArray):
    def __init__(
        self,
        antenna_dimension: Sequence[int] | torch.Tensor,
        antenna_spacing: Sequence[float] | torch.Tensor = [0.5, 0.5, 0.5],  # in unit of meter
        wavelength: float = 1.0,
    ) -> None:
        """Uniform 3D Cubic antenna array

        :param antenna_dimension: Sequence of number of antenna array elements along three dimensions
        :type antenna_dimension: Sequence[int] | torch.Tensor
        :param antenna_spacing: Sequence of antenna spacings along three dimensions in unit of meter
        :type antenna_spacing: Sequence[float] | torch.Tensor, optional, defaults to [0.5, 0.5, 0.5]
        """
        if len(antenna_dimension) != 3 or len(antenna_spacing) != 3:
            raise ValueError("Antenna Dimensions and/or Antenna Spacings are not three-dimensional")

        n_x, n_y, n_z = antenna_dimension[0], antenna_dimension[1], antenna_dimension[2]
        d_x, d_y, d_z = antenna_spacing[0], antenna_spacing[1], antenna_spacing[2]

        self.antenna_spacing_3d = antenna_spacing

        self.element_locations_x = -d_x * torch.arange(n_x)
        self.element_locations_y = -d_y * torch.arange(n_y)
        self.element_locations_z = -d_z * torch.arange(n_z)
        self.element_locations_grid_x, self.element_locations_grid_y, self.element_locations_grid_z = torch.meshgrid(
            self.element_locations_x,
            self.element_locations_y,
            self.element_locations_z,
            indexing="xy",
        )

        num_antennas = torch.prod(torch.as_tensor(antenna_dimension))
        element_locations = torch.zeros((num_antennas, 3))
        element_locations[:, 0] = self.element_locations_grid_x.flatten()
        element_locations[:, 1] = self.element_locations_grid_y.flatten()
        element_locations[:, 2] = self.element_locations_grid_z.flatten()

        super().__init__(element_locations=element_locations, wavelength=wavelength)
        assert self.num_antennas == num_antennas

    def steering_vector(
        self,
        angle: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        angle = format_angle(angle)
        svx = uniform_sv(
            element_locations=self.element_locations_x,
            angle=angle,
            wavelength=self.wavelength,
            mode="x",
        )
        svy = uniform_sv(
            element_locations=self.element_locations_y,
            angle=angle,
            wavelength=self.wavelength,
            mode="y",
        )
        svz = uniform_sv(
            element_locations=self.element_locations_z,
            angle=angle,
            wavelength=self.wavelength,
            mode="z",
        )
        return kron3(svy, svx, svz)


class UniformLinearArray(UniformCubicArray):
    def __init__(
        self,
        num_antennas: int,
        antenna_spacing: float = 0.5,  # in unit of meter
        along_axis: Literal["x", "y", "z"] = "z",
        wavelength: float = 1.0,
    ) -> None:
        """uniform linear antenna array (ULA) along a specified axis, in which antennas are placed in a line
            with the same spacing distance
        :param num_antennas: number of antennas along the specified axis
        :type num_antennas: int
        :param antenna_spacing: in unit of meter, default to 0.5
        :type antenna_spacing: float, optional
        :param along_axis: along which axis the array is placed, default to "z"
        :type along_axis: Literal['x', 'y', 'z'], optional
        :param wavelength: length of wave, default to 1.0
        :type wavelength: float, optional
        """

        assert along_axis in ["x", "y", "z"]
        self.along_axis = along_axis
        if along_axis == "x":
            antenna_dimension = (num_antennas, 1, 1)
            antenna_spacing_3d = (antenna_spacing, 0.0, 0.0)
        elif along_axis == "y":
            antenna_dimension = (1, num_antennas, 1)
            antenna_spacing_3d = (0.0, antenna_spacing, 0.0)
        else:
            antenna_dimension = (1, 1, num_antennas)
            antenna_spacing_3d = (0.0, 0.0, antenna_spacing)

        super().__init__(antenna_dimension=antenna_dimension, antenna_spacing=antenna_spacing_3d, wavelength=wavelength)
        self.antenna_spacing = antenna_spacing


class UniformRectangularArray(UniformCubicArray):
    def __init__(
        self,
        antenna_dimension: Sequence[int] | torch.Tensor,
        antenna_spacing: Sequence[float] | torch.Tensor = [0.5, 0.5],  # in unit of meter
        along_plane: Literal["xy", "yz", "xz"] = "yz",
        wavelength: float = 1.0,
    ) -> None:
        """uniform rectangular antenna array (URA) along a specified plane,
            in which antennas are placed in a rectangular shaped plane.
        :param antenna_dimension: Sequence of number of antenna array elements along two dimensions
        :type antenna_dimension: Sequence[int] | torch.Tensor
        :param antenna_spacing: Sequence of antenna spacings along two dimensions in unit of meter
        :type antenna_spacing: Sequence[float] | torch.Tensor, optional, defaults to [0.5, 0.5]
        :param along_plane: along which plane the array is placed, defaults to "yz"
        :type along_plane: Literal['xy', 'yz', 'xz'], optional
        :param wavelength: length of wave, defaults to 1.0
        :type wavelength: float, optional
        """

        # I should change this to a size check with antenna_dimensions
        if len(antenna_dimension) != 2 or len(antenna_spacing) != 2:
            raise ValueError("Antenna Dimensions and/or Antenna Spacings are not two-dimensional")

        assert along_plane in ["xy", "yz", "xz"]
        self.along_plane = along_plane
        if along_plane == "xy":
            antenna_dimension = (antenna_dimension[0], antenna_dimension[1], 1)
            antenna_spacing_3d = (antenna_spacing[0], antenna_spacing[1], 0.0)
        elif along_plane == "yz":
            antenna_dimension = (1, antenna_dimension[0], antenna_dimension[1])
            antenna_spacing_3d = (0.0, antenna_spacing[0], antenna_spacing[1])
        else:
            antenna_dimension = (antenna_dimension[0], 1, antenna_dimension[1])
            antenna_spacing_3d = (antenna_spacing[0], 0.0, antenna_spacing[1])

        super().__init__(antenna_dimension=antenna_dimension, antenna_spacing=antenna_spacing_3d, wavelength=wavelength)
        self.antenna_spacing = antenna_spacing


def uniform_sv(
    element_locations: torch.Tensor,
    angle: torch.Tensor,
    wavelength: float = 1,
    mode: Literal["x", "y", "z"] = "z",
):
    cos_, sin_ = torch.cos(angle), torch.sin(angle)
    cos_az, cos_el = cos_
    sin_az, sin_el = sin_
    if mode == "z":
        tmp = sin_el
    elif mode == "y":
        tmp = sin_az * cos_el
    else:
        tmp = cos_az * cos_el

    return exp1j2pi(tmp * element_locations / wavelength)
