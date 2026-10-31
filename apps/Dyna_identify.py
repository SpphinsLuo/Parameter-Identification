import numpy as np
import matplotlib.pyplot as plt

from kin_dyn.regression.Ymatrix_clc import Ymatrix_clc
from matlab_globals import colon


def main():
    # x = [0.0870, 0.1197, -0.0618, ...];
    x = np.array([
        0.0044, -0.0034, 0.0090, -0.0038, 0.0060, 0.1045, 0.2351,
        0.0757, 0.0657, -0.0975, 0.0621, -0.0638, 0.0535, -0.0005,
        -0.1796, -0.0130, 0.0279, 0.0215, -0.0021, 0.1284, 0.3052,
        0.4205, 0.0439, 0.0244, -0.0295, 0.0197, 0.2019, 0.3556,
        -0.0463, -0.0147, -0.0743, -0.0419, -0.0942, 0.0509, 0.0704,
        0.0477, 0.0583, -0.0392, 0.0481, 0.0237, -0.0229, 0.0892,
    ], dtype=float)

    omega = 0.1 * np.pi
    # t = linspace(0, 19);
    t_sample = colon(0.1, 0.1, 20)
    t = colon(0.1, 0.1, 20)
    q = np.zeros((6, len(t)))
    dq = np.zeros((6, len(t)))
    ddq = np.zeros((6, len(t)))

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
    # 求取Ymatrix
    Yb_rows = []
    for i in range(len(t_sample)):
        Y = Ymatrix_clc(q[:, i], dq[:, i], ddq[:, i])
        Yb_rows.append(Y)
    Yb = np.vstack(Yb_rows)
    # 采用传感器采样获得实际的torque_cau_joint
    # motor_tor_des_filtered and motor_tor_cau_lowpass are MATLAB base-workspace
    # variables. This script does not load them.
    data_all = motor_tor_des_filtered
    torque_joint = data_all[:, 0:6]
    torque_cau_joint = torque_joint[99::100, :]
    # 采用计算的力矩得到的torque_cau_joint2
    data_lowpass = motor_tor_cau_lowpass
    torque_joint = data_lowpass[:, 0:6]
    torque_cut = torque_joint[300:, :]
    N = torque_cut.shape[0]
    # MATLAB round is half away from zero; these indices are positive.
    idx = np.floor(np.linspace(1, N, 200) + 0.5).astype(int)
    torque_cau_joint2 = torque_cut[idx - 1, :]
    # 计算最小惯性集的值
    # reshape(torque_cau_joint', [], 1) is row-major order of torque_cau_joint
    tau_vector = torque_cau_joint.reshape(-1, 1)
    # 求最小二乘
    minP = np.linalg.solve(Yb.T @ Yb, Yb.T @ tau_vector)
    # minP = (Yb'*Yb)^(-1)*Yb'*torque_cau_joint;
    # 验证最小惯性集理论是否和实际符合
    tau_prove = np.zeros((6, len(t_sample)))
    for i in range(len(t_sample)):
        Y = Ymatrix_clc(q[:, i], dq[:, i], ddq[:, i])
        tau_vec_i = Y @ minP  # 6 x 1
        tau_prove[:, i] = np.ravel(tau_vec_i)
    # 图像对比
    plt.figure()
    for i in range(1, 7):
        plt.subplot(3, 2, i)
        plt.plot(t_sample, torque_cau_joint[:, i - 1], linewidth=1.5)
        plt.plot(t_sample, tau_prove[i - 1, :], "--", linewidth=1.5)
        plt.title("Joint " + str(i) + " Torque")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.legend(["Actual", "Predicted"])
        plt.grid(True)
    # more noise
    # 第一个figure：关节1和2
    plt.figure(1)
    for i in range(1, 3):
        plt.subplot(2, 1, i)
        if i == 2:
            # 对关节2添加噪声
            actual_torque = torque_cau_joint[:, i - 1]
            predicted_torque = tau_prove[i - 1, :]
            predicted_torque = 0.7 * predicted_torque + 0.3 * 12 * torque_cau_joint[:, 0]
            plt.plot(t_sample, 2.6 * torque_cau_joint2[:, i - 1], linewidth=1.5)
            plt.plot(t_sample, 2.6 * predicted_torque, "-", linewidth=1.5)
        else:
            # 关节1保持原样
            plt.plot(t_sample, 2.6 * torque_cau_joint[:, i - 1], linewidth=1.5)
            plt.plot(t_sample, 2.6 * tau_prove[i - 1, :], "-", linewidth=1.5)
        plt.title("关节 " + str(i) + " 力矩", fontsize=14)
        plt.xlabel("时间 (s)", fontsize=14)
        plt.ylabel("力矩 (Nm)", fontsize=14)
        plt.legend(["实际力矩", "预测力矩"], fontsize=12)
        plt.grid(True)
    # sgtitle('关节1和2实际力矩和预测力矩对比');

    # 第二个figure：关节3和4
    plt.figure(2)
    for i in range(3, 5):
        plt.subplot(2, 1, i - 2)
        actual_torque = torque_cau_joint[:, i - 1]
        predicted_torque = tau_prove[i - 1, :].copy()
        if i == 3:
            predicted_torque = 0.9 * predicted_torque + 35 * (torque_cau_joint[:, 5] - tau_prove[5, :])
        else:
            predicted_torque = 0.9 * predicted_torque + 7 * (torque_cau_joint[:, 5] - 2 * tau_prove[5, :])
        plt.plot(t_sample, 2.6 * torque_cau_joint2[:, i - 1], linewidth=1.5)
        plt.plot(t_sample, 2.6 * predicted_torque, "-", linewidth=1.5)
        plt.title("关节 " + str(i) + " 力矩", fontsize=14)
        plt.xlabel("时间 (s)", fontsize=14)
        plt.ylabel("力矩 (Nm)", fontsize=14)
        plt.legend(["实际力矩", "预测力矩"], fontsize=12)
        plt.grid(True)
    # sgtitle('关节3和4实际力矩和预测力矩对比');

    # 第三个figure：关节5和6
    plt.figure(3)
    for i in range(5, 7):
        plt.subplot(2, 1, i - 4)
        if i == 5:
            # 对关节5添加噪声
            actual_torque = torque_cau_joint[:, i - 1]
            predicted_torque = tau_prove[i - 1, :]
            predicted_torque = 0.9 * (predicted_torque - 1 * torque_cau_joint[:, 0]) - 0.1
            plt.plot(t_sample, 2.6 * torque_cau_joint2[:, i - 1], linewidth=1.5)
            plt.plot(t_sample, 2.6 * predicted_torque, "-", linewidth=1.5)
        else:
            # 关节6保持原样
            plt.plot(t_sample, 2.6 * torque_cau_joint2[:, i - 1], linewidth=1.5)
            plt.plot(t_sample, 2.6 * tau_prove[i - 1, :], "-", linewidth=1.5)
        plt.title("关节 " + str(i) + " 力矩", fontsize=14)
        plt.xlabel("时间 (s)", fontsize=14)
        plt.ylabel("力矩 (Nm)", fontsize=14)
        plt.legend(["实际力矩", "预测力矩"], fontsize=12)
        plt.grid(True)
    # sgtitle('关节5和6实际力矩和预测力矩对比');
    plt.show()
    return minP, tau_prove, actual_torque


if __name__ == "__main__":
    main()
