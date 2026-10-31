import numpy as np


def getK(vec3):
    v = np.asarray(vec3, dtype=float).reshape(-1, order="F")
    K = np.array([
        [v[0], v[1], v[2], 0.0, 0.0, 0.0],
        [0.0, v[0], 0.0, v[1], v[2], 0.0],
        [0.0, 0.0, v[0], 0.0, v[1], v[2]],
    ])
    return K

def getS(vec3):
    v = np.asarray(vec3, dtype=float).reshape(-1, order="F")
    S = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return S

def Vec2so3(vec):
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 1:
        row, col = arr.shape[0], 1
        v = arr
    else:
        row, col = arr.shape
        v = arr.reshape(-1, order="F")
    if row != 3 and col != 1:
        raise ValueError("worng,input of Vec2so3 must be 3*1 vector")
    screw = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return screw
