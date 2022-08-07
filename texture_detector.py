from typing import Optional
from typing import Union
from typing import List

import numpy as np
import cv2

from config import Config
from pre_proc.base_preprocessor import BasePreprocessor
from pre_proc.preprocessor import DefaultPreprocessor
from post_proc.base_postprocessor import BasePostprocessor
from post_proc.postprocessor import DefaultPostprocessor
from dct.image_dct import ImageDCT


class TexDet:
    def __init__(self,
                 config: Config,
                 postprocessor: Union[BasePostprocessor, None] = None,
                 preprocessors: Union[List[BasePreprocessor], BasePreprocessor, None] = None):
        self.config = config

        if postprocessor is None:
            self.post = DefaultPostprocessor()
        else:
            self.post = postprocessor

        if preprocessors is None:
            self.pre = [DefaultPreprocessor()]
        elif isinstance(preprocessors, BasePreprocessor):
            self.pre = [preprocessors]
        else:
            self.pre = preprocessors

        self.dct = ImageDCT(config)
        self.cells = None

    def run(self, path_or_list: Union[List[str], str]) -> np.ndarray:
        if isinstance(path_or_list, list):
            pass
        elif isinstance(path_or_list, str):
            return self._proc_img(path_or_list)

    def _proc_img(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        for preprop in self.pre:
            img = preprop.run(img)

        self.cells = self.dct.run_dct(img)
        scores = np.zeros(self.config.calc_cell_alignment(img.shape))

        for cell_i in range(self.cells.shape[0]):
            for cell_j in range(self.cells.shape[1]):
                scores[cell_i, cell_j] = self.post.detect(self.cells[cell_i, cell_j])
        return scores
