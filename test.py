import numpy as np

from MDHTrans import MDHTrans
from getAi import getAi
from getHi import getHi
from getInertia import getInertia
from get_Yf_Yn import get_Yf_Yn
from matlab_globals import G
from motion_para_clc import motion_para_clc
from tau_newtonEuler import tau_newtonEuler


def main():
    # q, dq, ddq are MATLAB base-workspace variables. They are not defined here.
    q_in = np.asarray(q, dtype=float).reshape(-1, order="F")
    dq_in = np.asarray(dq, dtype=float).reshape(-1, order="F")
    ddq_in = np.asarray(ddq, dtype=float).reshape(-1, order="F")
    q1 = q_in[0]
    q2 = q_in[1]
    q3 = q_in[2]
    q4 = q_in[3]
    q5 = q_in[4]
    q6 = q_in[5]
    q7 = q_in[6]
    q8 = q_in[7]
    dq1 = dq_in[0]
    dq2 = dq_in[1]
    dq3 = dq_in[2]
    dq4 = dq_in[3]
    dq5 = dq_in[4]
    dq6 = dq_in[5]
    dq7 = dq_in[6]
    dq8 = dq_in[7]
    ddq1 = ddq_in[0]
    ddq2 = ddq_in[1]
    ddq3 = ddq_in[2]
    ddq4 = ddq_in[3]
    ddq5 = ddq_in[4]
    ddq6 = ddq_in[5]
    ddq7 = ddq_in[6]
    ddq8 = ddq_in[7]
    g = 9.8
    Pc11 = np.array([0.0, 0.0, 0.0])
    Pc22 = np.array([0.0, 0.0, 0.0])
    Pc33 = np.array([0.0, 0.0, 0.0])
    Pc44 = np.array([0.0, 0.0, 0.0])
    Pc55 = np.array([0.0, 0.0, 0.0])
    Pc66 = np.array([0.0, 0.0, 0.0])
    Pc77 = np.array([0.0, 0.0, 0.0])
    Pc88 = np.array([0.0, 0.0, 0.0])
    G["g"] = g
    G["Pc11"] = Pc11
    G["Pc22"] = Pc22
    G["Pc33"] = Pc33
    G["Pc44"] = Pc44
    G["Pc55"] = Pc55
    G["Pc66"] = Pc66
    G["Pc77"] = Pc77
    G["Pc88"] = Pc88
    # 坐标系旋转轴
    axis1 = np.array([0.0, 0.0, 1.0])
    axis2 = np.array([0.0, 0.0, 1.0])
    axis3 = np.array([0.0, 0.0, 1.0])
    axis4 = np.array([0.0, 0.0, 1.0])
    axis5 = np.array([0.0, 0.0, 1.0])
    axis6 = np.array([0.0, 0.0, 1.0])
    axis7 = np.array([0.0, 0.0, 1.0])
    axis8 = np.array([0.0, 0.0, 1.0])
    # 坐标系变换矩阵 alpha a d theta
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
    qr = np.array([q1, q2, q3, q4, q5, q6, q7, q8, 0.0])
    # qr=[q1, q2, q3, q4, q5, q6, q7]
    DH_q0 = MDH[:, 3]
    q_dh = (qr + DH_q0) * np.pi / 180
    alpha = MDH[:, 0]
    a = MDH[:, 1]
    d = MDH[:, 2]

    T01 = MDHTrans(alpha[0], a[0], d[0], q_dh[0])
    T12 = MDHTrans(alpha[1], a[1], d[1], q_dh[1])
    T23 = MDHTrans(alpha[2], a[2], d[2], q_dh[2])
    T34 = MDHTrans(alpha[3], a[3], d[3], q_dh[3])
    T45 = MDHTrans(alpha[4], a[4], d[4], q_dh[4])
    T56 = MDHTrans(alpha[5], a[5], d[5], q_dh[5])
    T67 = MDHTrans(alpha[6], a[6], d[6], q_dh[6])
    T78 = MDHTrans(alpha[7], a[7], d[7], q_dh[7])
    T89 = MDHTrans(alpha[8], a[8], d[8], q_dh[8])

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
    w77, dw77, dv77, dvc77 = motion_para_clc(R76, dq7, ddq7, w66, dw66, dv66, P67, Pc77, axis7)
    w88, dw88, dv88, dvc88 = motion_para_clc(R87, dq8, ddq8, w77, dw77, dv77, P78, Pc88, axis8)
    # 辅助矩阵H
    H1 = getHi(w11, dw11, dv11)
    H2 = getHi(w22, dw22, dv22)
    H3 = getHi(w33, dw33, dv33)
    H4 = getHi(w44, dw44, dv44)
    H5 = getHi(w55, dw55, dv55)
    H6 = getHi(w66, dw66, dv66)
    H7 = getHi(w77, dw77, dv77)
    H8 = getHi(w88, dw88, dv88)
    # 辅助矩阵A
    A1 = getAi(w11, dw11, dv11)
    A2 = getAi(w22, dw22, dv22)
    A3 = getAi(w33, dw33, dv33)
    A4 = getAi(w44, dw44, dv44)
    A5 = getAi(w55, dw55, dv55)
    A6 = getAi(w66, dw66, dv66)
    A7 = getAi(w77, dw77, dv77)
    A8 = getAi(w88, dw88, dv88)
    # 辅助矩阵Yf、Yn
    # get_Yf_Yn is written for 6 links (3x60). The source still calls it with
    # an 8-link regressor (3x80), which does not conform.
    Yf8 = np.hstack([np.zeros((3, 70)), H8])
    Yn8 = np.hstack([np.zeros((3, 70)), A8])
    Yf7, Yn7 = get_Yf_Yn(7, R78, H7, A7, Yf8, Yn8, P78)
    Yf6, Yn6 = get_Yf_Yn(6, R67, H6, A6, Yf7, Yn7, P67)
    Yf5, Yn5 = get_Yf_Yn(5, R56, H5, A5, Yf6, Yn6, P56)
    Yf4, Yn4 = get_Yf_Yn(4, R45, H4, A4, Yf5, Yn5, P45)
    Yf3, Yn3 = get_Yf_Yn(3, R34, H3, A3, Yf4, Yn4, P34)
    Yf2, Yn2 = get_Yf_Yn(2, R23, H2, A2, Yf3, Yn3, P23)
    Yf1, Yn1 = get_Yf_Yn(1, R12, H1, A1, Yf2, Yn2, P12)
    # 线性矩阵Y
    Yr = np.zeros((8, 80))
    Yr[0, 0:80] = axis1 @ Yn1
    Yr[1, 0:80] = axis2 @ Yn2
    Yr[2, 0:80] = axis3 @ Yn3
    Yr[3, 0:80] = axis4 @ Yn4
    Yr[4, 0:80] = axis5 @ Yn5
    Yr[5, 0:80] = axis6 @ Yn6
    # Source uses axis6 for rows 7 and 8, not axis7/axis8.
    Yr[6, 0:80] = axis6 @ Yn7
    Yr[7, 0:80] = axis6 @ Yn8
    # 考虑摩擦力的情况下
    Yff = np.zeros((8, 2))
    Yff[0, :] = [np.sign(dq1), dq1]
    Yff[1, :] = [np.sign(dq2), dq2]
    Yff[2, :] = [np.sign(dq3), dq3]
    Yff[3, :] = [np.sign(dq4), dq4]
    Yff[4, :] = [np.sign(dq5), dq5]
    Yff[5, :] = [np.sign(dq6), dq6]
    Yff[6, :] = [np.sign(dq7), dq7]
    Yff[7, :] = [np.sign(dq8), dq8]
    Y1 = np.hstack([Yr, Yff])
    Y = np.zeros((8, 72))
    for i in range(1, 9):
        Y[:, (9 * i - 8) - 1] = Yr[:, (10 * i - 9) - 1]  # xx
        Y[:, (9 * i - 7) - 1] = Yr[:, (10 * i - 8) - 1]  # xy
        Y[:, (9 * i - 6) - 1] = Yr[:, (10 * i - 7) - 1]  # xz
        Y[:, (9 * i - 5) - 1] = Yr[:, (10 * i - 5) - 1]  # yz
        Y[:, (9 * i - 4) - 1] = Yr[:, (10 * i - 4) - 1]  # zz
        Y[:, (9 * i - 3) - 1] = Yr[:, (10 * i - 3) - 1]  # mx
        Y[:, (9 * i - 2) - 1] = Yr[:, (10 * i - 2) - 1]  # my
        Y[:, (9 * i - 1) - 1] = np.sign(dq_in[i - 1])  # fc
        Y[:, (9 * i) - 1] = dq_in[i - 1]  # fv
    Y22 = np.zeros((8, 56))
    for i in range(1, 9):
        Y22[:, (7 * i - 6) - 1:7 * i] = Y[:, (9 * i - 8) - 1:(9 * i - 2)]

    Pr = np.ones((80, 1))
    fc = np.ones((8, 1))
    fv = np.ones((8, 1))
    P = []
    for i in range(8):
        outs = getInertia(1, 1, 1, 1, 1, 1, 1, 1, 1, a[i], d[i], alpha[i], 1, 1, 1, 1)
        XX_i, XY_i, XZ_i, YZ_i, ZZ_i, MX_i, MY_i = outs[:7]
        P.extend([XX_i, XY_i, XZ_i, YZ_i, ZZ_i, MX_i, MY_i])
    P = np.asarray(P, dtype=float)
    tau_newton = tau_newtonEuler(1, 1, 1)
    tao_r = Yr @ Pr
    tao = Y22 @ P.reshape(-1, 1)
    del R98, P89, dvc00, Y1, fc, fv, axis7, axis8, ddq1
    return Y, Y22, tau_newton, tao_r, tao


if __name__ == "__main__":
    main()
