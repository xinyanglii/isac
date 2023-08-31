from abc import ABC

import torch


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
