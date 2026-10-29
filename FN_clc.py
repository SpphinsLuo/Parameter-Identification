import numpy as np


def FN_clc(w, dw, dvc, m, Ic):
    Fc = m * dvc
    Nc = Ic @ dw + np.cross(w, Ic @ w)
    return Fc, Nc
