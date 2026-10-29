import numpy as np


def getK(vec3):
    v = np.asarray(vec3, dtype=float).reshape(-1, order="F")
    K = np.array([
        [v[0], v[1], v[2], 0.0, 0.0, 0.0],
        [0.0, v[0], 0.0, v[1], v[2], 0.0],
        [0.0, 0.0, v[0], 0.0, v[1], v[2]],
    ])
    return K
