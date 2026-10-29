import numpy as np


def getS(vec3):
    v = np.asarray(vec3, dtype=float).reshape(-1, order="F")
    S = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return S
