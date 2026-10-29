import numpy as np


def Fourier(q0, a, b, t, n):
    y = q0
    dy = 0
    ddy = 0
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    for i in range(1, n + 1):
        y = y + a[i - 1] * np.cos(i * t) + b[i - 1] * np.sin(i * t)
        dy = dy - i * a[i - 1] * np.sin(i * t) + i * b[i - 1] * np.cos(i * t)
        ddy = ddy - i ** 2 * a[i - 1] * np.cos(i * t) - i ** 2 * b[i - 1] * np.sin(i * t)
    return y, dy, ddy
