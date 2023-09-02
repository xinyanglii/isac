from __future__ import annotations

from abc import ABC
from typing import Literal, Sequence, Union

import torch

from ..utils import exp1j2pi, kron3
from .angle_utils import format_angle


class AbstractAntennaArray(ABC):
    def __init__(
        self,
        element_locations: torch.Tensor,
        wavelength: float = 1,
    ) -> None:
        r"""Abstract base antenna array class

        :param element_locations: locations of antenna elements in unit of meter, of
                                shape (num_antennas, 3)
        :type element_locations: torch.Tensor
        :param wavelength: wavelength of the signal, defaults to 1
        :type wavelength: float, optional
        """
        self.element_locations = element_locations
        self.wavelength = wavelength

    @property
    def element_locations(self) -> torch.Tensor:
        return self._element_locations

    @element_locations.setter
    def element_locations(self, el_loc: torch.Tensor) -> None:
        assert len(el_loc.shape) == 2 and el_loc.shape[1] == 3
        self._element_locations = el_loc

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl: float) -> None:
        assert wl > 0
        self._wavelength = wl

    @property
    def num_antennas(self) -> int:
        return len(self.element_locations)


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
        grid: Sequence[float] | torch.Tensor,
        axis: int = 0,
    ) -> torch.Tensor:
        """Generate steering matrix given angle grid and stacking axis

        :param grid: Grid of angles on which steering vectors are computed, grid[i] is the i-the angle,
        :type grid: Sequence[float] | torch.Tensor
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
        :type antenna_spacing: Sequence[float] | torch.Tensor, defaults to [0.5, 0.5, 0.5]
        """
        if len(antenna_dimension) != 3 or len(antenna_spacing) != 3:
            raise ValueError("Antenna Dimensions and/or Antenna Spacings are not three-dimensional")

        n_x, n_y, n_z = antenna_dimension[0], antenna_dimension[1], antenna_dimension[2]
        d_x, d_y, d_z = antenna_spacing[0], antenna_spacing[1], antenna_spacing[2]

        self.antenna_spacing_3d = antenna_spacing
        self.antenna_spacing = antenna_spacing

        self.element_locations_x = -d_x * torch.arange(n_x)
        self.element_locations_y = -d_y * torch.arange(n_y)
        self.element_locations_z = -d_z * torch.arange(n_z)
        self.element_locations_grid_x, self.element_locations_grid_y, self.element_locations_grid_z = torch.meshgrid(
            self.element_locations_x,
            self.element_locations_y,
            self.element_locations_z,
            indexing="xy",
        )

        element_locations = torch.stack(
            (
                self.element_locations_grid_x.flatten(),
                self.element_locations_grid_y.flatten(),
                self.element_locations_grid_z.flatten(),
            ),
            dim=1,
        )

        super().__init__(element_locations=element_locations, wavelength=wavelength)
        assert self.num_antennas == n_x * n_y * n_z

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
        :type antenna_spacing: Sequence[float] | torch.Tensor, defaults to [0.5, 0.5]
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
