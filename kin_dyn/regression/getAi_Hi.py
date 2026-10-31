import numpy as np

from kin_dyn.regression.math import getK, getS


def getAi(wi, dwi, dvi):
    Ai = np.zeros((3, 10))
    Ai[:, 0:6] = getK(dwi) + getS(wi) @ getK(wi)
    Ai[:, 6:9] = -getS(dvi)
    Ai[:, 9] = 0.0
    return Ai

def getHi(wi, dwi, dvi):
    Hi = np.zeros((3, 10))
    Hi[:, 6:9] = getS(dwi) + getS(wi) @ getS(wi)
    Hi[:, 9] = np.asarray(dvi, dtype=float).reshape(-1, order="F")
    Hi[:, 0:6] = 0.0
    return Hi