import numpy as np
import matplotlib.pyplot as plt

from MDHTrans import MDHTrans
from matlab_globals import colon


def main():
    # x is a MATLAB base-workspace variable (Fourier coefficients). It is not
    # defined in this script.
    omega = 0.1 * np.pi
    t = colon(0, 1, 2 * np.pi / omega - 1)  # 采样点
    qr = np.zeros((8, len(t)))
    dq = np.zeros((8, len(t)))
    ddq = np.zeros((8, len(t)))
    x_coeff = np.asarray(x, dtype=float).reshape(-1, order="F")
    p = None
    for i in range(1, 9):
        if i < 4:
            p = x_coeff[(i * 7 - 6) - 1:i * 7]
        if i == 4:
            qr[3, :] = qr[2, :]
            dq[3, :] = dq[2, :]
            ddq[3, :] = ddq[2, :]
        if i > 4:
            p = x_coeff[((i - 1) * 7 - 6) - 1:(i - 1) * 7]
        qr[i - 1, :] = (p[0] + p[1] * np.sin(omega * t) + p[2] * np.cos(omega * t)
                        + p[3] * np.sin(2 * omega * t) + p[4] * np.cos(2 * omega * t)
                        + p[5] * np.sin(3 * omega * t) + p[6] * np.cos(3 * omega * t))
        dq[i - 1, :] = (p[1] * omega * np.cos(omega * t) - p[2] * omega * np.sin(omega * t)
                        + p[3] * 2 * omega * np.cos(2 * omega * t) - p[4] * 2 * omega * np.sin(2 * omega * t)
                        + p[5] * 3 * omega * np.cos(3 * omega * t) - p[6] * 3 * omega * np.sin(3 * omega * t))
        ddq[i - 1, :] = (-p[1] * omega ** 2 * np.sin(omega * t) - p[2] * omega ** 2 * np.cos(omega * t)
                         - p[3] * 4 * omega ** 2 * np.sin(2 * omega * t) - p[4] * 4 * omega ** 2 * np.cos(2 * omega * t)
                         - p[5] * 9 * omega ** 2 * np.sin(3 * omega * t) - p[6] * 9 * omega ** 2 * np.cos(3 * omega * t))
    MDH = np.array([
        [-90, 0, 0, 0],
        [-90, 0, 0, -90],
        [-90, 0, 0, -90],
        [0, 112, 0, 0],
        [0, 360, 0, 0],
        [0, 360, 0, 0],
        [-90, 0, 0, 180],
        [-90, 42.5, 0, -90],
        [-90, 60, -64, -90],
    ], dtype=float)
    # qr=[q1, q2, q3, q4, q5, q6, q7 ,q8 ,0]
    # qr=[q1, q2, q3, q4, q5, q6, q7]
    DH_q0 = MDH[:, 3] * 3.14 / 180
    q = np.zeros((9, len(t)))
    for i in range(1, 9):
        q[i - 1, :] = qr[i - 1, :] + DH_q0[i - 1]
    q[8, :] = 0
    alpha = MDH[:, 0]
    a = MDH[:, 1]
    d = MDH[:, 2]

    x11 = np.zeros(len(t))
    y11 = np.zeros(len(t))
    z11 = np.zeros(len(t))
    for i in range(len(t)):
        T01 = MDHTrans(alpha[0], a[0], d[0], q[0, i])
        T12 = MDHTrans(alpha[1], a[1], d[1], q[1, i])
        T23 = MDHTrans(alpha[2], a[2], d[2], q[2, i])
        T34 = MDHTrans(alpha[3], a[3], d[3], q[3, i])
        T45 = MDHTrans(alpha[4], a[4], d[4], q[4, i])
        T56 = MDHTrans(alpha[5], a[5], d[5], q[5, i])
        T67 = MDHTrans(alpha[6], a[6], d[6], q[6, i])
        T78 = MDHTrans(alpha[7], a[7], d[7], q[7, i])
        T89 = MDHTrans(alpha[8], a[8], d[8], q[8, i])

        R01 = T01[0:3, 0:3]
        R10 = R01.T
        P01 = T01[0:3, 3]
        R12 = T12[0:3, 0:3]
        R21 = R12.T
        P12 = T12[0:3, 3]
        R23 = T23[0:3, 0:3]
        R32 = R23.T
        P23 = T23[0:3, 3]
        R34 = T34[0:3, 0:3]
        R43 = R34.T
        P34 = T34[0:3, 3]
        R45 = T45[0:3, 0:3]
        R54 = R45.T
        P45 = T45[0:3, 3]
        R56 = T56[0:3, 0:3]
        R65 = R56.T
        P56 = T56[0:3, 3]
        R67 = T67[0:3, 0:3]
        R76 = R67.T
        P67 = T67[0:3, 3]
        R78 = T78[0:3, 0:3]
        R87 = R78.T
        P78 = T78[0:3, 3]
        R89 = T89[0:3, 0:3]
        R98 = R89.T
        P89 = T89[0:3, 3]

        T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T78 @ T89
        x11[i] = T[0, 3]  # x 轨迹
        y11[i] = T[1, 3]  # y 轨迹
        z11[i] = T[2, 3]
        del R10, P01, R21, P12, R32, P23, R43, P34, R54, P45, R65, P56, R76, P67, R87, P78, R98, P89

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x11, y11, z11, linewidth=2)
    plt.show()
    return q, dq, ddq


if __name__ == "__main__":
    main()
