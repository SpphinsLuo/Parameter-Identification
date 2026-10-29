import numpy as np


def Rightleg(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p00 = -0.0003017
    p10 = 1.006
    p01 = -0.921
    p20 = 0.01826
    p11 = -0.3303
    p02 = -0.09775
    p30 = 0.009589
    p21 = -0.04277
    p12 = 0.326
    p03 = 0.1752
    # Source splits this polynomial across two physical lines without '...';
    # both lines are one expression.
    q_motor = (p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2
               + p30 * x ** 3 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3)
    return q_motor
