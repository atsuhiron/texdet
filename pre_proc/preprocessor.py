import numpy as np

from pre_proc.base_preprocessor import BasePreprocessor

import cv2


class DefaultPreprocessor(BasePreprocessor):
    def run(self, data: np.ndarray) -> np.ndarray:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return data
