from typing import Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    dct_data_type: type = np.float32
    cell_size: int = 32

    def calc_cell_alignment(self, arr_shape: Tuple[int, int]) -> Tuple[int, int]:
        return arr_shape[0] // self.cell_size, arr_shape[1] // self.cell_size


@dataclass
class StripPPConfig:
    sigma: float
    std_thresh: float
    min_pattern_freq_index: int
    """
    検出するテクスチャパターンの最大スケール。セル全体のの長さに対して、`1 / min_pattern_freq_index` より大きいパターンを無視する。
    """

    max_pattern_freq_index: int
    """
    検出するテクスチャパターンの最小スケール。セル全体のの長さに対して、`1 / max_pattern_freq_index` より小さいパターンを無視する。
    """

