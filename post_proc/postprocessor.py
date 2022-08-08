from typing import Tuple
import numpy as np
import scipy.ndimage as sn

from config import StripPPConfig
from post_proc.base_postprocessor import BasePostprocessor


class DefaultPostprocessor(BasePostprocessor):
    def __init__(self):
        pass

    def detect(self, cell: np.ndarray) -> float:
        return float(np.max(cell))


class StripPP(BasePostprocessor):
    def __init__(self, spp_config: StripPPConfig):
        self.spp_config = spp_config
        self.prof_slice = slice(self.spp_config.min_pattern_freq_index, self.spp_config.max_pattern_freq_index)

    def detect(self, cell: np.ndarray) -> float:
        vertical_res = self._detect(cell, 0)
        horizontal_res = self._detect(cell, 1)
        return np.linalg.norm(np.array([vertical_res[0], horizontal_res[0]]))

    def _detect(self, cell: np.ndarray, axis: int) -> Tuple[float, int]:
        prof = self.calc_smoothed_profile(cell, axis)
        search_res = self._try_search_first_peak_index(prof)
        if not search_res[0]:
            return 0.0, -1

        peak_val = prof[search_res[1]]

        peak_sigma = self._calc_peak_sigma(peak_val, prof.mean(), prof.std())
        return peak_sigma, search_res[1]

    def calc_smoothed_profile(self, data: np.ndarray, axis: int) -> np.ndarray:
        prof = np.mean(data, axis=axis)[self.prof_slice]
        if self.spp_config.sigma <= 0:
            return prof
        prof = sn.gaussian_filter1d(prof, self.spp_config.sigma, mode="constant", cval=0.0)
        return prof

    def _try_search_first_peak_index(self, profile: np.ndarray) -> Tuple[bool, int]:
        sharpness = np.zeros(len(profile), dtype=np.float64)
        for i in range(1, len(profile) - 1):
            _prev = profile[i-1]
            _curr = profile[i]
            _next = profile[i+1]
            sharpness[i] = 2 * _curr / (_prev + _next)

        if np.max(sharpness) > 0:
            return True, int(np.argmax(sharpness))
        return False, -1

    @staticmethod
    def _calc_peak_sigma(x: float, mean: float, std: float) -> float:
        return (x - mean) / std
