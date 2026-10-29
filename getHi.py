import numpy as np

from getS import getS


def getHi(wi, dwi, dvi):
    Hi = np.zeros((3, 10))
    Hi[:, 6:9] = getS(dwi) + getS(wi) @ getS(wi)
    Hi[:, 9] = np.asarray(dvi, dtype=float).reshape(-1, order="F")
    Hi[:, 0:6] = 0.0
    return Hi
