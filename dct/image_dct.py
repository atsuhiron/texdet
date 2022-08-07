from typing import Tuple

import numpy as np
import scipy.fft as fft

from config import Config


class ImageDCT:
    def __init__(self, config: Config):
        self.config = config

    def run_dct(self, arr: np.ndarray) -> np.ndarray:
        c_size = self.config.cell_size
        cell_alignment = self.config.calc_cell_alignment(arr.shape)
        out_size = (cell_alignment[0], cell_alignment[1], c_size, c_size)
        cell_arr = np.zeros(out_size, dtype=self.config.dct_data_type)

        for cell_i in range(out_size[0]):
            for cell_j in range(out_size[1]):
                slice_i = slice(cell_i * c_size, (cell_i + 1) * c_size)
                slice_j = slice(cell_j * c_size, (cell_j + 1) * c_size)
                cell = arr[slice_i, slice_j]
                dct_cell = self._dct_2d(cell)
                cell_arr[cell_i, cell_j] = dct_cell
        return cell_arr

    def _dct_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(dtype=self.config.dct_data_type)
        arr -= np.mean(arr)
        vertical = fft.dct(arr, axis=0)
        horizontal = fft.dct(arr, axis=1)
        return np.sqrt(vertical*vertical + horizontal*horizontal)

