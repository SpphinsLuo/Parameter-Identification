import numpy as np

from MDHTrans import MDHTrans
from getAi import getAi
from getHi import getHi
from get_Yf_Yn import get_Yf_Yn
from matlab_globals import G
from motion_para_clc import motion_para_clc


# 不含摩擦的 6×60 完整回归矩阵，QR 分解用这个
def Ymatrix_withoutff(q, dq, ddq):
    q = np.asarray(q, dtype=float).reshape(-1, order="F")
    dq = np.asarray(dq, dtype=float).reshape(-1, order="F")
    ddq = np.asarray(ddq, dtype=float).reshape(-1, order="F")
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    dq1 = dq[0]
    dq2 = dq[1]
    dq3 = dq[2]
    dq4 = dq[3]
    dq5 = dq[4]
    dq6 = dq[5]
    ddq1 = ddq[0]
    ddq2 = ddq[1]
    ddq3 = ddq[2]
    ddq4 = ddq[3]
    ddq5 = ddq[4]
    ddq6 = ddq[5]
    g = 9.8
    Pc11 = np.array([1.0, 1.0, 1.0])
    Pc22 = np.array([1.0, 1.0, 1.0])
    Pc33 = np.array([1.0, 1.0, 1.0])
    Pc44 = np.array([1.0, 1.0, 1.0])
    Pc55 = np.array([1.0, 1.0, 1.0])
    Pc66 = np.array([1.0, 1.0, 1.0])
    G["Pc11"] = Pc11
    G["Pc22"] = Pc22
    G["Pc33"] = Pc33
    G["Pc44"] = Pc44
    G["Pc55"] = Pc55
    G["Pc66"] = Pc66

    # 坐标系旋转轴
    axis1 = np.array([0.0, 0.0, 1.0])
    axis2 = np.array([0.0, 0.0, 1.0])
    axis3 = np.array([0.0, 0.0, 1.0])
    axis4 = np.array([0.0, 0.0, 1.0])
    axis5 = np.array([0.0, 0.0, 1.0])
    axis6 = np.array([0.0, 0.0, 1.0])

    # 坐标系变换矩阵 alpha a d theta
    MDH = np.array([
        [0, 0, -269, 0],
        [90, 0, 0, 90],
        [90, 0, 0, 180],
        [0, 360, 0, 0],
        [0, 360, 0, 0],
        [90, 0, 0, 0],
        [0, 79, 0, 0],
    ], dtype=float)
    qr = np.array([q1, q2, q3, q4, q5, q6, 0.0])
    # qr=[q1, q2, q3, q4, q5, q6, q7]
    DH_q0 = MDH[:, 3]
    q = qr + DH_q0 * np.pi / 180
    alpha = MDH[:, 0] * np.pi / 180
    a = 0.001 * MDH[:, 1]
    d = 0.001 * MDH[:, 2]

    T01 = MDHTrans(alpha[0], a[0], d[0], q[0])
    T12 = MDHTrans(alpha[1], a[1], d[1], q[1])
    T23 = MDHTrans(alpha[2], a[2], d[2], q[2])
    T34 = MDHTrans(alpha[3], a[3], d[3], q[3])
    T45 = MDHTrans(alpha[4], a[4], d[4], q[4])
    T56 = MDHTrans(alpha[5], a[5], d[5], q[5])
    T67 = MDHTrans(alpha[6], a[6], d[6], q[6])

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

    # 外推
    w00 = np.array([0.0, 0.0, 0.0])
    dw00 = np.array([0.0, 0.0, 0.0])
    dv00 = np.array([0.0, 0.0, -g])
    dvc00 = np.array([0.0, 0.0, -g])
    w11, dw11, dv11, dvc11 = motion_para_clc(R10, dq1, ddq1, w00, dw00, dv00, P01, Pc11, axis1)
    w22, dw22, dv22, dvc22 = motion_para_clc(R21, dq2, ddq2, w11, dw11, dv11, P12, Pc22, axis2)
    w33, dw33, dv33, dvc33 = motion_para_clc(R32, dq3, ddq3, w22, dw22, dv22, P23, Pc33, axis3)
    w44, dw44, dv44, dvc44 = motion_para_clc(R43, dq4, ddq4, w33, dw33, dv33, P34, Pc44, axis4)
    w55, dw55, dv55, dvc55 = motion_para_clc(R54, dq5, ddq5, w44, dw44, dv44, P45, Pc55, axis5)
    w66, dw66, dv66, dvc66 = motion_para_clc(R65, dq6, ddq6, w55, dw55, dv55, P56, Pc66, axis6)

    # 辅助矩阵H
    H1 = getHi(w11, dw11, dv11)
    H2 = getHi(w22, dw22, dv22)
    H3 = getHi(w33, dw33, dv33)
    H4 = getHi(w44, dw44, dv44)
    H5 = getHi(w55, dw55, dv55)
    H6 = getHi(w66, dw66, dv66)

    # 辅助矩阵A
    A1 = getAi(w11, dw11, dv11)
    A2 = getAi(w22, dw22, dv22)
    A3 = getAi(w33, dw33, dv33)
    A4 = getAi(w44, dw44, dv44)
    A5 = getAi(w55, dw55, dv55)
    A6 = getAi(w66, dw66, dv66)

    # 辅助矩阵Yf、Yn
    Yf6 = np.hstack([np.zeros((3, 50)), H6])
    Yn6 = np.hstack([np.zeros((3, 50)), A6])
    Yf5, Yn5 = get_Yf_Yn(5, R56, H5, A5, Yf6, Yn6, P56)
    Yf4, Yn4 = get_Yf_Yn(4, R45, H4, A4, Yf5, Yn5, P45)
    Yf3, Yn3 = get_Yf_Yn(3, R34, H3, A3, Yf4, Yn4, P34)
    Yf2, Yn2 = get_Yf_Yn(2, R23, H2, A2, Yf3, Yn3, P23)
    Yf1, Yn1 = get_Yf_Yn(1, R12, H1, A1, Yf2, Yn2, P12)
    # 线性矩阵Y
    Yr = np.zeros((6, 60))
    Yr[0, 0:60] = axis1 @ Yn1
    Yr[1, 0:60] = axis2 @ Yn2
    Yr[2, 0:60] = axis3 @ Yn3
    Yr[3, 0:60] = axis4 @ Yn4
    Yr[4, 0:60] = axis5 @ Yn5
    Yr[5, 0:60] = axis6 @ Yn6
    Y = Yr
    del R76, P67, dvc00, dvc11, dvc22, dvc33, dvc44, dvc55, dvc66, Yf1
    return Y
