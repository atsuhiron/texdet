import abc

import numpy as np


class BasePostprocessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(self, cell: np.ndarray) -> float:
        pass
