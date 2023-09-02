import numpy as np
import torch
from scipy.signal import find_peaks as sp_find_peaks

from isac.utils import find_peaks


def test_find_peaks(device):
    x = np.random.randn(100)
    x_torch = torch.from_numpy(x).to(device)
    _, pidx = find_peaks(x_torch)
    sp_pidx, _ = sp_find_peaks(x)
    assert all([p.item() in sp_pidx for p in pidx])
