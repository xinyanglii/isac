import pytest
import torch

from isac.mimo import Generic3DAntennaArray, UniformCubicArray, UniformLinearArray, UniformRectangularArray
from isac.utils import freq2wavelen, uniform


def assert_against_generic(arr: Generic3DAntennaArray, axis, wavelength, angle, grid):
    arr_gen = Generic3DAntennaArray(element_locations=arr.element_locations, wavelength=wavelength)

    stv = arr.steering_vector(angle=angle)
    stv_gen = arr_gen.steering_vector(angle=angle)

    assert stv.shape == stv_gen.shape
    assert torch.allclose(stv, stv_gen, atol=1e-6)

    stm = arr.steering_matrix(grid=grid, axis=axis)
    stm_gen = arr_gen.steering_matrix(grid=grid, axis=axis)

    assert stm.shape == stm_gen.shape
    assert torch.allclose(stm, stm_gen)


@pytest.mark.parametrize("along_axis", ["x", "y", "z"])
def test_uniform_linear_array(
    along_axis,
    carrier_frequency=2.4e9,
    num_ant_tx=8,
    d=0.5,
    axis=0,
    n_grid=10,
):
    fc = carrier_frequency
    wavelength = freq2wavelen(fc)
    num_ants = num_ant_tx
    ant_spacing = d * wavelength
    angle = uniform(-torch.pi / 2, torch.pi / 2)
    grid = uniform(-torch.pi / 2, torch.pi / 2, n_grid)

    arr = UniformLinearArray(
        num_antennas=num_ants,
        antenna_spacing=ant_spacing,
        along_axis=along_axis,
        wavelength=wavelength,
    )
    assert_against_generic(arr, axis=axis, wavelength=wavelength, angle=angle, grid=grid)


@pytest.mark.parametrize("along_plane", ["xy", "yz", "xz"])
def test_uniform_rectangular_array(
    along_plane,
    carrier_frequency=2.4e9,
    num_ant_tx=8,
    num_ant_rx=8,
    d1=0.5,
    d2=0.3,
    axis=0,
    n_grid: int = 10,
):
    num_ant_y = num_ant_tx
    num_ant_z = num_ant_rx
    fc, antenna_dimension, d = carrier_frequency, [num_ant_y, num_ant_z], [d1, d2]
    wavelength = freq2wavelen(fc)
    ant_spacing = [x * wavelength for x in d]

    az = uniform(-torch.pi, torch.pi)
    el = uniform(-torch.pi / 2, torch.pi / 2)
    angle = torch.as_tensor([az, el])

    az_grid = torch.deg2rad(uniform(-180, 180, size=n_grid))
    el_grid = torch.deg2rad(uniform(-90, 90, size=n_grid))
    grid = torch.stack([az_grid, el_grid], dim=1)

    arr = UniformRectangularArray(
        antenna_dimension=antenna_dimension,
        antenna_spacing=ant_spacing,
        along_plane=along_plane,
        wavelength=wavelength,
    )

    assert_against_generic(arr, axis=axis, wavelength=wavelength, angle=angle, grid=grid)


def test_uniform_cubic_array(
    carrier_frequency=2.4e9,
    num_ant_tx=8,
    num_ant_rx=8,
    num_ant_z=8,
    d1=0.5,
    d2=0.2,
    d3=0.3,
    axis=0,
    n_grid=10,
):
    num_ant_x = num_ant_tx
    num_ant_y = num_ant_rx
    fc, antenna_dimension, d = carrier_frequency, [num_ant_x, num_ant_y, num_ant_z], [d1, d2, d3]
    wavelength = freq2wavelen(fc)
    ant_spacing = [x * wavelength for x in d]

    az = torch.deg2rad(uniform(-180, 180))
    el = torch.deg2rad(uniform(-90, 90))
    angle = torch.as_tensor([az, el])

    az_grid = torch.deg2rad(uniform(-180, 180, size=n_grid))
    el_grid = torch.deg2rad(uniform(-90, 90, size=n_grid))
    grid = torch.stack([az_grid, el_grid], dim=1)

    arr = UniformCubicArray(antenna_dimension=antenna_dimension, antenna_spacing=ant_spacing, wavelength=wavelength)

    assert_against_generic(arr, axis=axis, wavelength=wavelength, angle=angle, grid=grid)
