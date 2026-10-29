import numpy as np


def MDHTrans(alpha, a, d, theta):
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    return T
