import numpy as np

from kin_dyn.FN_clc import FN_clc
from kin_dyn.MDHTrans import MDHTrans
from kin_dyn.regression.Ymatrix_withoutff import Ymatrix_withoutff
from kin_dyn.fn import fn
from kin_dyn.regression.getAi_Hi import getAi, getHi
from kin_dyn.getInertia import getInertia
from kin_dyn.regression.get_Yf_Yn import get_Yf_Yn
from matlab_globals import G
from kin_dyn.motion_para_clc import motion_para_clc


def main():
    # q=30*3.14/180*ones(8,1);dq=45*3.14/180*ones(8,1);ddq=45*3.14/180*ones(8,1);
    # clear; clc
    q_start = np.random.rand(6, 20)
    dq_start = np.random.rand(6, 20)
    ddq_start = np.random.rand(6, 20)
    # q_start = ones(6,20);dq_start=ones(6,20);ddq_start=ones(6,20);
    q1 = 0
    q2 = 0.2
    q3 = -0.3
    q4 = 0.7
    q5 = -0.5
    q6 = 0

    dq1 = dq_start[0, 0]
    dq2 = dq_start[1, 0]
    dq3 = dq_start[2, 0]
    dq4 = dq_start[3, 0]
    dq5 = dq_start[4, 0]
    dq6 = dq_start[5, 0]

    ddq1 = ddq_start[0, 0]
    ddq2 = ddq_start[1, 0]
    ddq3 = ddq_start[2, 0]
    ddq4 = ddq_start[3, 0]
    ddq5 = ddq_start[4, 0]
    ddq6 = ddq_start[5, 0]
    Ic1 = np.diag([1.0, 1.0, 1.0])
    Ic2 = np.diag([1.0, 1.0, 1.0])
    Ic3 = np.diag([1.0, 1.0, 1.0])
    Ic4 = np.diag([1.0, 1.0, 1.0])
    Ic5 = np.diag([1.0, 1.0, 1.0])
    Ic6 = np.diag([1.0, 1.0, 1.0])
    Pc11 = np.array([1.0, 1.0, 1.0])
    Pc22 = np.array([1.0, 1.0, 1.0])
    Pc33 = np.array([1.0, 1.0, 1.0])
    Pc44 = np.array([1.0, 1.0, 1.0])
    Pc55 = np.array([1.0, 1.0, 1.0])
    Pc66 = np.array([1.0, 1.0, 1.0])
    m1 = 2
    m2 = 2
    m3 = 2
    m4 = 2
    m5 = 2
    m6 = 2
    g = 9.8
    G.update(dict(
        m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, m6=m6, g=g,
        Ic1=Ic1, Ic2=Ic2, Ic3=Ic3, Ic4=Ic4, Ic5=Ic5, Ic6=Ic6,
        Pc11=Pc11, Pc22=Pc22, Pc33=Pc33, Pc44=Pc44, Pc55=Pc55, Pc66=Pc66,
    ))
    # fc*, fv* are declared global in the source and never assigned.
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
    ], dtype=float)  # 最后一行为到足底的距离
    q_deg = np.array([q1, q2, q3, q4, q5, q6, 0.0])
    # qr=[q1, q2, q3, q4, q5, q6, q7]
    DH_q0 = MDH[:, 3]
    q = q_deg + DH_q0 * np.pi / 180
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
    TTT = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67

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

    # 外推：连杆角速度、角加速度、线加速度、质心加速度、质心受力
    w00 = np.array([0.0, 0.0, 0.0])
    dw00 = np.array([0.0, 0.0, 0.0])
    dv00 = np.array([0.0, 0.0, -g])
    dvc00 = np.array([0.0, 0.0, -g])  # g还是-g
    w11, dw11, dv11, dvc11 = motion_para_clc(R10, dq1, ddq1, w00, dw00, dv00, P01, Pc11, axis1)
    w22, dw22, dv22, dvc22 = motion_para_clc(R21, dq2, ddq2, w11, dw11, dv11, P12, Pc22, axis2)
    w33, dw33, dv33, dvc33 = motion_para_clc(R32, dq3, ddq3, w22, dw22, dv22, P23, Pc33, axis3)
    w44, dw44, dv44, dvc44 = motion_para_clc(R43, dq4, ddq4, w33, dw33, dv33, P34, Pc44, axis4)
    w55, dw55, dv55, dvc55 = motion_para_clc(R54, dq5, ddq5, w44, dw44, dv44, P45, Pc55, axis5)
    w66, dw66, dv66, dvc66 = motion_para_clc(R65, dq6, ddq6, w55, dw55, dv55, P56, Pc66, axis6)

    # Y矩阵分析
    H1 = getHi(w11, dw11, dv11)
    H2 = getHi(w22, dw22, dv22)
    H3 = getHi(w33, dw33, dv33)
    H4 = getHi(w44, dw44, dv44)
    H5 = getHi(w55, dw55, dv55)
    H6 = getHi(w66, dw66, dv66)

    A1 = getAi(w11, dw11, dv11)
    A2 = getAi(w22, dw22, dv22)
    A3 = getAi(w33, dw33, dv33)
    A4 = getAi(w44, dw44, dv44)
    A5 = getAi(w55, dw55, dv55)
    A6 = getAi(w66, dw66, dv66)

    Yf6 = np.hstack([np.zeros((3, 50)), H6])
    Yn6 = np.hstack([np.zeros((3, 50)), A6])
    Yf5, Yn5 = get_Yf_Yn(5, R56, H5, A5, Yf6, Yn6, P56)
    Yf4, Yn4 = get_Yf_Yn(4, R45, H4, A4, Yf5, Yn5, P45)
    Yf3, Yn3 = get_Yf_Yn(3, R34, H3, A3, Yf4, Yn4, P34)
    Yf2, Yn2 = get_Yf_Yn(2, R23, H2, A2, Yf3, Yn3, P23)
    Yf1, Yn1 = get_Yf_Yn(1, R12, H1, A1, Yf2, Yn2, P12)

    Yr = np.zeros((6, 60))
    Yr[0, 0:60] = axis1 @ Yn1
    Yr[1, 0:60] = axis2 @ Yn2
    Yr[2, 0:60] = axis3 @ Yn3
    Yr[3, 0:60] = axis4 @ Yn4
    Yr[4, 0:60] = axis5 @ Yn5
    Yr[5, 0:60] = axis6 @ Yn6

    # 考虑摩擦力的情况下
    Yff = np.zeros((6, 2))
    Yff[0, :] = [np.sign(dq1), dq1]
    Yff[1, :] = [np.sign(dq2), dq2]
    Yff[2, :] = [np.sign(dq3), dq3]
    Yff[3, :] = [np.sign(dq4), dq4]
    Yff[4, :] = [np.sign(dq5), dq5]
    Yff[5, :] = [np.sign(dq6), dq6]

    Fc11, Nc11 = FN_clc(w11, dw11, dvc11, m1, Ic1)
    Fc22, Nc22 = FN_clc(w22, dw22, dvc22, m2, Ic2)
    Fc33, Nc33 = FN_clc(w33, dw33, dvc33, m3, Ic3)
    Fc44, Nc44 = FN_clc(w44, dw44, dvc44, m4, Ic4)
    Fc55, Nc55 = FN_clc(w55, dw55, dvc55, m5, Ic5)
    Fc66, Nc66 = FN_clc(w66, dw66, dvc66, m6, Ic6)
    # 内推：连杆受力、关节力矩   非线性化，惯性张量的坐标在质心->tau=n'*axis;
    f77 = np.array([0.0, 0.0, 0.0])
    n77 = np.array([0.0, 0.0, 0.0])
    f66, n66 = fn(R67, f77, Fc66, Nc66, n77, Pc66, P67)
    f55, n55 = fn(R56, f66, Fc55, Nc55, n66, Pc55, P56)
    f44, n44 = fn(R45, f55, Fc44, Nc44, n55, Pc44, P45)
    f33, n33 = fn(R34, f44, Fc33, Nc33, n44, Pc33, P34)
    f22, n22 = fn(R23, f33, Fc22, Nc22, n33, Pc22, P23)
    f11, n11 = fn(R12, f22, Fc11, Nc11, n22, Pc11, P12)
    tau_newtown = np.array([
        n11 @ axis1, n22 @ axis2, n33 @ axis3, n44 @ axis4, n55 @ axis5, n66 @ axis6,
    ]).reshape(-1, 1)

    # 采用最小惯性集的方法
    row = np.array([[1.0, 1.0, 1.0]])
    Io = np.diag([1.0, 1.0, 1.0]) + 2 * ((row @ row.T) * np.diag([1.0, 1.0, 1.0]) - (row.T @ row))  # 3,3,3,-1,-1,-1
    P_base = np.array([5, -2, -2, 5, -2, 5, 2, 2, 2, 2], dtype=float)
    Pr = np.tile(P_base, 6)
    fc = np.ones((6, 1))
    fv = np.ones((6, 1))
    XX_ = np.zeros(6)
    XY_ = np.zeros(6)
    XZ_ = np.zeros(6)
    YY_ = np.zeros(6)
    YZ_ = np.zeros(6)
    ZZ_ = np.zeros(6)
    MX_ = np.zeros(6)
    MY_ = np.zeros(6)
    MZ_ = np.zeros(6)
    M_ = np.zeros(6)
    (XX_[5], XY_[5], XZ_[5], YY_[5], YZ_[5], ZZ_[5], MX_[5], MY_[5], MZ_[5], M_[5]) = getInertia(
        5, 5, 5, -2, -2, -2, 1, 1, 1, a[6], d[6], alpha[6], 0, 0, 0, 2)
    (XX_[4], XY_[4], XZ_[4], YY_[4], YZ_[4], ZZ_[4], MX_[4], MY_[4], MZ_[4], M_[4]) = getInertia(
        5, 5, 5, -2, -2, -2, 1, 1, 1, a[5], d[5], alpha[5], YY_[5], MZ_[5], M_[5], 2)
    (XX_[3], XY_[3], XZ_[3], YY_[3], YZ_[3], ZZ_[3], MX_[3], MY_[3], MZ_[3], M_[3]) = getInertia(
        5, 5, 5, -2, -2, -2, 1, 1, 1, a[4], d[4], alpha[4], YY_[4], MZ_[4], M_[4], 2)
    (XX_[2], XY_[2], XZ_[2], YY_[2], YZ_[2], ZZ_[2], MX_[2], MY_[2], MZ_[2], M_[2]) = getInertia(
        5, 5, 5, -2, -2, -2, 1, 1, 1, a[3], d[3], alpha[3], YY_[3], MZ_[3], M_[3], 2)
    (XX_[1], XY_[1], XZ_[1], YY_[1], YZ_[1], ZZ_[1], MX_[1], MY_[1], MZ_[1], M_[1]) = getInertia(
        5, 5, 5, -2, -2, -2, 1, 1, 1, a[2], d[2], alpha[2], YY_[2], MZ_[2], M_[2], 2)
    (XX_[0], XY_[0], XZ_[0], YY_[0], YZ_[0], ZZ_[0], MX_[0], MY_[0], MZ_[0], M_[0]) = getInertia(
        5, 5, 5, -2, -2, -2, 1, 1, 1, a[1], d[1], alpha[1], YY_[1], MZ_[1], M_[1], 2)
    P = []
    for i in range(6):
        P.extend([XX_[i], XY_[i], XZ_[i], YZ_[i], ZZ_[i], MX_[i], MY_[i]])
    P = np.asarray(P, dtype=float)
    # QR分解方法
    # Yr_20 is 20*8 in the comment; each Ymatrix_withoutff call appends 6 rows.
    blocks = []
    for i in range(20):
        Yr_rand = Ymatrix_withoutff(q_start[:, i], dq_start[:, i], ddq_start[:, i])
        blocks.append(Yr_rand)
    Yr_20 = np.vstack(blocks)
    Q, R = np.linalg.qr(Yr_20, mode="reduced")
    wuguan = []
    youguan = []
    for i in range(R.shape[1]):
        if abs(R[i, i]) < 1e-4:
            wuguan.append(i + 1)
        else:
            youguan.append(i + 1)

    if youguan:
        YQR1 = np.hstack([Yr_20[:, num - 1:num] for num in youguan])
    else:
        YQR1 = np.empty((Yr_20.shape[0], 0))
    if wuguan:
        YQR2 = np.hstack([Yr_20[:, num - 1:num] for num in wuguan])
    else:
        YQR2 = np.empty((Yr_20.shape[0], 0))

    youguan_num = YQR1.shape[1]
    Q_r, R_r = np.linalg.qr(np.hstack([YQR1, YQR2]), mode="reduced")
    R1 = R_r[0:youguan_num, 0:youguan_num]
    R2 = R_r[0:youguan_num, youguan_num:]
    beta = np.linalg.inv(R1) @ R2
    test = YQR2 - YQR1 @ beta
    # handle the influence of computational accuracy
    # for i=1:size(beta,1)
    #     for j=1:size(beta,2)
    #      if(abs(beta(i,j))<1e-5)
    #          beta(i,j)=0;
    #      end
    #     end
    # end
    base_param = []
    minset_cols = []
    for n in youguan:
        base_param.append(Pr[n - 1])
        minset_cols.append(Yr[:, n - 1:n])
    base_param = np.asarray(base_param, dtype=float)
    minset_Y = np.hstack(minset_cols) if minset_cols else np.empty((Yr.shape[0], 0))
    none_base_param = []
    nonset_cols = []
    for n in wuguan:
        none_base_param.append(Pr[n - 1])
        nonset_cols.append(Yr[:, n - 1:n])
    none_base_param = np.asarray(none_base_param, dtype=float)
    nonset_Y = np.hstack(nonset_cols) if nonset_cols else np.empty((Yr.shape[0], 0))
    minsetP = base_param.reshape(-1, 1) + beta @ none_base_param.reshape(-1, 1)
    # 力矩输出
    tao_r = Yr @ Pr.reshape(-1, 1)
    # tao_r_jianhua = Y22*P'
    tao_QR = minset_Y @ minsetP
    # err = tau_newtown-tao_r_jianhua;
    # 验证
    # Pmin = inv(Y22'*Y22)*Y22'*tao_r_jianhua;
    del TTT, R76, dvc00, Yf1, Yff, Io, fc, fv, Q, Q_r, test, nonset_Y, P
    return tau_newtown, tao_r, tao_QR, minsetP


if __name__ == "__main__":
    main()
