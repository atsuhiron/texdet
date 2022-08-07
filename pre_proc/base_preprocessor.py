import abc

import numpy as np


class BasePreprocessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, data: np.ndarray) -> np.ndarray:
        pass
