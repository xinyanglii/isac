import pytest
import torch

from isac.antenna.array_utils import format_angle


def test_format_angle():
    # testing with correct input leaves everything unchanged.
    angle_scalar = 0.1
    angle_list = [0.0, 0.1]

    angle = torch.as_tensor(angle_list)

    res_scalar = format_angle(angle_scalar)
    res_t = format_angle(angle)

    assert -torch.pi <= res_t[0] <= torch.pi  # azimuth
    assert -torch.pi / 2 <= res_t[1] <= torch.pi / 2  # elevation
    assert torch.allclose(res_scalar, res_t)

    # test with input out of valid angle region, which should be wrapped
    angle_valid = torch.as_tensor([0.1, 0.1])
    angle_invalid = angle_valid + torch.as_tensor([2 * torch.pi, 2 * torch.pi])
    with pytest.warns(UserWarning):
        res_valid = format_angle(angle_invalid)
    assert torch.allclose(res_valid, angle_valid)

    # if the elevation angle is still out of [-pi/2, pi/2] after wrapping, all angles should be transformed geometrically
    angle_invalid[0] = angle_valid[0] - torch.pi * 2
    angle_invalid[1] = torch.pi - angle_valid[1]
    with pytest.warns(UserWarning):
        res_valid = format_angle(angle_invalid)
    # elevation angle does not change but azimuth angle transformed by pi
    assert torch.isclose(angle_valid[0] - torch.pi, res_valid[0])
    assert torch.isclose(angle_valid[1], res_valid[1])

    # test with wrong output
    # if the assertion errors are not raised test is failed
    angle_complex = torch.as_tensor([1j, 0.0])
    angle_list_of_lists = torch.as_tensor([[0.0, 0.0]])

    for a in (angle_complex, angle_list_of_lists):
        with pytest.raises(AssertionError):
            format_angle(a)
