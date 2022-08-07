import cv2

from texture_detector import TexDet
from config import Config
from config import StripPPConfig
import graphix
from post_proc.postprocessor import StripPP

if __name__ == "__main__":
    ipath = "image/sample.png"
    conf = Config()
    spp_config = StripPPConfig(
        sigma=0.1,
        std_thresh=1.0,
        min_pattern_freq_index=4,
        max_pattern_freq_index=20
    )
    spp = StripPP(spp_config)
    texdet = TexDet(conf, postprocessor=spp)
    score = texdet.run(ipath)

    img = cv2.cvtColor(cv2.imread(ipath), cv2.COLOR_BGR2RGB)
    graphix.show_overlay_score(img, score, conf)



    v = spp.calc_smoothed_profile(texdet.cells[2, 12], 0)
    h = spp.calc_smoothed_profile(texdet.cells[2, 12], 1)
    graphix.show_profile(v, h, spp.prof_slice)
