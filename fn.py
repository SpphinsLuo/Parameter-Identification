import numpy as np


def fn(R, f_next, Fc, Nc, n_next, Pc, Po):
    f = R @ f_next + Fc
    n = Nc + R @ n_next + np.cross(Pc, Fc) + np.cross(Po, R @ f_next)
    return f, n
