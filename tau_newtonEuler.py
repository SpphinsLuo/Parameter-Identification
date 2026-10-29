import numpy as np

from FN_clc import FN_clc
from MDHTrans import MDHTrans
from fn import fn
from matlab_globals import G
from motion_para_clc import motion_para_clc


def tau_newtonEuler(q, dq, ddq):
    q = np.asarray(q, dtype=float).reshape(-1, order="F")
    dq = np.asarray(dq, dtype=float).reshape(-1, order="F")
    ddq = np.asarray(ddq, dtype=float).reshape(-1, order="F")
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]
    q8 = q[7]
    dq1 = dq[0]
    dq2 = dq[1]
    dq3 = dq[2]
    dq4 = dq[3]
    dq5 = dq[4]
    dq6 = dq[5]
    dq7 = dq[6]
    dq8 = dq[7]
    ddq1 = ddq[0]
    ddq2 = ddq[1]
    ddq3 = ddq[2]
    ddq4 = ddq[3]
    ddq5 = ddq[4]
    ddq6 = ddq[5]
    ddq7 = ddq[6]
    ddq8 = ddq[7]
    # MATLAB globals. m*, Ic*, Pc*, g must already have been stored by the caller.
    # fc* and fv* are declared global in the source and are only used in comments.
    m1 = G["m1"]
    m2 = G["m2"]
    m3 = G["m3"]
    m4 = G["m4"]
    m5 = G["m5"]
    m6 = G["m6"]
    m7 = G["m7"]
    m8 = G["m8"]
    g = G["g"]
    Ic1 = G["Ic1"]
    Ic2 = G["Ic2"]
    Ic3 = G["Ic3"]
    Ic4 = G["Ic4"]
    Ic5 = G["Ic5"]
    Ic6 = G["Ic6"]
    Ic7 = G["Ic7"]
    Ic8 = G["Ic8"]
    Pc11 = np.asarray(G["Pc11"], dtype=float).reshape(3)
    Pc22 = np.asarray(G["Pc22"], dtype=float).reshape(3)
    Pc33 = np.asarray(G["Pc33"], dtype=float).reshape(3)
    Pc44 = np.asarray(G["Pc44"], dtype=float).reshape(3)
    Pc55 = np.asarray(G["Pc55"], dtype=float).reshape(3)
    Pc66 = np.asarray(G["Pc66"], dtype=float).reshape(3)
    Pc77 = np.asarray(G["Pc77"], dtype=float).reshape(3)
    Pc88 = np.asarray(G["Pc88"], dtype=float).reshape(3)
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
    qr = np.array([q1, q2, q3, q4, q5, q6, q7, q8])
    # qr=[q1, q2, q3, q4, q5, q6, q7]
    # The source assigns DH_q0 from DH, which is not defined in this function
    # (the table above is named MDH). The NameError is the original behavior.
    DH_q0 = DH[:, 3]
    q = (qr + DH_q0) * np.pi / 180
    alpha = DH[:, 0]
    a = DH[:, 1]
    d = DH[:, 2]

    T01 = MDHTrans(alpha[0], a[0], d[0], q[0])
    T12 = MDHTrans(alpha[1], a[1], d[1], q[1])
    T23 = MDHTrans(alpha[2], a[2], d[2], q[2])
    T34 = MDHTrans(alpha[3], a[3], d[3], q[3])
    T45 = MDHTrans(alpha[4], a[4], d[4], q[4])
    T56 = MDHTrans(alpha[5], a[5], d[5], q[5])
    T67 = MDHTrans(alpha[6], a[6], d[6], q[6])
    T78 = MDHTrans(alpha[7], a[7], d[7], q[7])
    T89 = MDHTrans(alpha[8], a[8], d[8], q[8])

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
    # 外推：连杆角速度、角加速度、线加速度、质心加速度、质心受力
    w00 = np.array([0.0, 0.0, 0.0])
    dw00 = np.array([0.0, 0.0, 0.0])
    dv00 = np.array([0.0, 0.0, g])
    dvc00 = np.array([0.0, 0.0, g])  # g还是-g
    w11, dw11, dv11, dvc11 = motion_para_clc(R10, dq1, ddq1, w00, dw00, dv00, P01, Pc11, axis1)
    w22, dw22, dv22, dvc22 = motion_para_clc(R21, dq2, ddq2, w11, dw11, dv11, P12, Pc22, axis2)
    w33, dw33, dv33, dvc33 = motion_para_clc(R32, dq3, ddq3, w22, dw22, dv22, P23, Pc33, axis3)
    w44, dw44, dv44, dvc44 = motion_para_clc(R43, dq4, ddq4, w33, dw33, dv33, P34, Pc44, axis4)
    w55, dw55, dv55, dvc55 = motion_para_clc(R54, dq5, ddq5, w44, dw44, dv44, P45, Pc55, axis5)
    w66, dw66, dv66, dvc66 = motion_para_clc(R65, dq6, ddq6, w55, dw55, dv55, P56, Pc66, axis6)
    w77, dw77, dv77, dvc77 = motion_para_clc(R76, dq7, ddq7, w66, dw66, dv66, P67, Pc77, axis7)
    w88, dw88, dv88, dvc88 = motion_para_clc(R87, dq8, ddq8, w77, dw77, dv77, P78, Pc88, axis8)
    Fc11, Nc11 = FN_clc(w11, dw11, dvc11, m1, Ic1)
    Fc22, Nc22 = FN_clc(w22, dw22, dvc22, m2, Ic2)
    Fc33, Nc33 = FN_clc(w33, dw33, dvc33, m3, Ic3)
    Fc44, Nc44 = FN_clc(w44, dw44, dvc44, m4, Ic4)
    Fc55, Nc55 = FN_clc(w55, dw55, dvc55, m5, Ic5)
    Fc66, Nc66 = FN_clc(w66, dw66, dvc66, m6, Ic6)
    Fc77, Nc77 = FN_clc(w77, dw77, dvc77, m7, Ic7)
    Fc88, Nc88 = FN_clc(w88, dw88, dvc88, m8, Ic8)
    # 内推：连杆受力、关节力矩   非线性化，惯性张量的坐标在质心->tau=n'*axis;
    f99 = np.array([0.0, 0.0, 0.0])
    n99 = np.array([0.0, 0.0, 0.0])
    # Nc99 is not defined in the source (Nc88 is). The NameError is original.
    f88, n88 = fn(R89, f99, Fc88, Nc99, n99, Pc88, P89)
    f77, n77 = fn(R78, f88, Fc77, Nc77, n88, Pc77, P78)
    f66, n66 = fn(R67, f77, Fc66, Nc66, n77, Pc66, P67)
    f55, n55 = fn(R56, f66, Fc55, Nc55, n66, Pc55, P56)
    f44, n44 = fn(R45, f55, Fc44, Nc44, n55, Pc44, P45)
    f33, n33 = fn(R34, f44, Fc33, Nc33, n44, Pc33, P34)
    f22, n22 = fn(R23, f33, Fc22, Nc22, n33, Pc22, P23)
    f11, n11 = fn(R12, f22, Fc11, Nc11, n22, Pc11, P12)
    tau = np.array([
        n11 @ axis1, n22 @ axis2, n33 @ axis3, n44 @ axis4,
        n55 @ axis5, n66 @ axis6, n77 @ axis7, n88 @ axis8,
    ])
    # 考虑摩擦力,膝盖1：1，踝关节3：1，趾关节2：1
    # C=C_clc(q);
    # ff = [fv1*dq1+fc1*sgn(dq1), ...]'
    # tau_f=C*tau+ff;
    del dvc00, R98, Fc88, Nc88
    return tau
