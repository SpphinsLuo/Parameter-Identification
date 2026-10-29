import numpy as np

from getK import getK
from getS import getS


def getAi(wi, dwi, dvi):
    Ai = np.zeros((3, 10))
    Ai[:, 0:6] = getK(dwi) + getS(wi) @ getK(wi)
    Ai[:, 6:9] = -getS(dvi)
    Ai[:, 9] = 0.0
    return Ai
