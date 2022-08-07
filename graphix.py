from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from config import Config


def show_overlay_score(img: np.ndarray, score: np.ndarray, config: Config):
    cell_mask = np.zeros(img.shape[0:2], dtype=np.float32)
    cell_alignment = config.calc_cell_alignment(img.shape)
    cell_mask[0: config.cell_size * cell_alignment[0], 0: config.cell_size * cell_alignment[1]] = 0.5

    score_arr = np.zeros(img.shape[0:2], dtype=np.float32)
    for cell_i in range(cell_alignment[0]):
        for cell_j in range(cell_alignment[1]):
            slice_i = slice(cell_i * config.cell_size, (cell_i + 1) * config.cell_size)
            slice_j = slice(cell_j * config.cell_size, (cell_j + 1) * config.cell_size)
            score_arr[slice_i, slice_j] = score[cell_i, cell_j]

    plt.imshow(img)
    plt.imshow(score_arr, cmap="jet", alpha=cell_mask)
    plt.show()


def show_profile(vert: np.ndarray, hori: np.ndarray, prof_slice: Optional[slice] = None):
    if prof_slice is None:
        x_range = np.arange(len(vert))
    else:
        x_range = np.arange(prof_slice.start, prof_slice.stop)

    plt.plot(x_range, vert, label="Vertical")
    plt.plot(x_range, hori, label="Horizontal")
    plt.legend()
    plt.show()
