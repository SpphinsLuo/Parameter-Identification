import numpy as np

from Ymatrix_clc import Ymatrix_clc
from matlab_globals import colon


def cond_aver(x):
    omega = 0.1 * np.pi
    t = colon(0, 1, 2 * np.pi / omega - 1)  # 采样点
    q = np.zeros((6, len(t)))
    dq = np.zeros((6, len(t)))
    ddq = np.zeros((6, len(t)))

    cond_aver_value = 0
    print("x")
    print(x)

    x = np.asarray(x, dtype=float).reshape(-1, order="F")
    for i in range(1, 7):
        p = x[(i * 7 - 6) - 1:i * 7]
        q[i - 1, :] = (p[0] + p[1] * np.sin(omega * t) + p[2] * np.cos(omega * t)
                       + p[3] * np.sin(2 * omega * t) + p[4] * np.cos(2 * omega * t)
                       + p[5] * np.sin(3 * omega * t) + p[6] * np.cos(3 * omega * t))
        dq[i - 1, :] = (p[1] * omega * np.cos(omega * t) - p[2] * omega * np.sin(omega * t)
                        + p[3] * 2 * omega * np.cos(2 * omega * t) - p[4] * 2 * omega * np.sin(2 * omega * t)
                        + p[5] * 3 * omega * np.cos(3 * omega * t) - p[6] * 3 * omega * np.sin(3 * omega * t))
        ddq[i - 1, :] = (-p[1] * omega ** 2 * np.sin(omega * t) - p[2] * omega ** 2 * np.cos(omega * t)
                         - p[3] * 4 * omega ** 2 * np.sin(2 * omega * t) - p[4] * 4 * omega ** 2 * np.cos(2 * omega * t)
                         - p[5] * 9 * omega ** 2 * np.sin(3 * omega * t) - p[6] * 9 * omega ** 2 * np.cos(3 * omega * t))

    if (np.max(np.abs(q[0, :])) - 15 * np.pi / 180 > 0
            or np.max(np.abs(q[1, :])) - 10 * np.pi / 180 > 0
            or np.max(np.abs(q[2, :] + 10 * np.pi / 180)) - 20 * np.pi / 180 > 0
            or np.max(np.abs(q[3, :] - 25 * np.pi / 180)) - 25 * np.pi / 180 > 0
            or np.max(np.abs(q[4, :])) - 10 * np.pi / 180 > 0
            or np.max(np.abs(q[5, :])) - 10 * np.pi / 180 > 0):  # 关节位置
        cond_aver_value = 1e5  # 关节限位

    for i in range(len(t)):
        # MATLAB passes the row transpose; q(k) is unchanged for a vector.
        B = Ymatrix_clc(q[:, i].T, dq[:, i].T, ddq[:, i].T)  # 计算当前关节的系数矩阵
        # B = Ymatrix_withoutff(q[:, i].T, dq[:, i].T, ddq[:, i].T)
        s_B = np.linalg.svd(B, compute_uv=False)
        for j in range(len(s_B)):
            if s_B[j] < 1e-4:
                s_B = s_B[:j]  # 去掉0特征值
                break
        cond_aver_value = cond_aver_value + np.max(s_B) / np.min(s_B)
        print(cond_aver_value)

    print("cond_aver")
    print(cond_aver_value)
    return cond_aver_value
