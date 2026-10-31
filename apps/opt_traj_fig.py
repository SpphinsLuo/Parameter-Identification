import os
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

from kin_dyn.j2m import Leftleg, Rightleg
from matlab_globals import colon


def main():
    omega = 0.1 * np.pi
    t = colon(0, 0.1, 2 * np.pi / omega - 1)  # 采样点
    q = np.zeros((6, len(t)))
    dq = np.zeros((6, len(t)))
    ddq = np.zeros((6, len(t)))
    q_left = np.zeros((6, len(t)))

    cond_aver = 0
    x = np.array([
        [0.0044, -0.0034, 0.0090, -0.0038, 0.0060, 0.1045, 0.2351],
        [0.0757, 0.0657, -0.0975, 0.0621, -0.0638, 0.0535, -0.0005],
        [-0.1796, -0.0130, 0.0279, 0.0215, -0.0021, 0.1284, 0.3052],
        [0.4205, 0.0439, 0.0244, -0.0295, 0.0197, 0.2019, 0.3556],
        [-0.0463, -0.0147, -0.0743, -0.0419, -0.0942, 0.0509, 0.0704],
        [0.0477, 0.0583, -0.0392, 0.0481, 0.0237, -0.0229, 0.0892],
    ], dtype=float)

    for i in range(1, 7):
        p = x[i - 1, :]
        q[i - 1, :] = (p[0] + p[1] * np.sin(omega * t) + p[2] * np.cos(omega * t)
                       + p[3] * np.sin(2 * omega * t) + p[4] * np.cos(2 * omega * t)
                       + p[5] * np.sin(3 * omega * t) + p[6] * np.cos(3 * omega * t))
        dq[i - 1, :] = (p[1] * omega * np.cos(omega * t) - p[2] * omega * np.sin(omega * t)
                        + p[3] * 2 * omega * np.cos(2 * omega * t) - p[4] * 2 * omega * np.sin(2 * omega * t)
                        + p[5] * 3 * omega * np.cos(3 * omega * t) - p[6] * 3 * omega * np.sin(3 * omega * t))
        ddq[i - 1, :] = (-p[1] * omega ** 2 * np.sin(omega * t) - p[2] * omega ** 2 * np.cos(omega * t)
                         - p[3] * 4 * omega ** 2 * np.sin(2 * omega * t) - p[4] * 4 * omega ** 2 * np.cos(2 * omega * t)
                         - p[5] * 9 * omega ** 2 * np.sin(3 * omega * t) - p[6] * 9 * omega ** 2 * np.cos(3 * omega * t))
    # 髋关节转换成关节角度
    q_left = Leftleg(q[1, :], q[2, :])
    q_right = Rightleg(q[1, :], q[2, :])
    # 绘制图像
    plt.figure(1)
    for i in range(1, 7):
        plt.plot(t, q[i - 1, :], label="Joint " + str(i))
    plt.title("Joint Displacement (q)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position(rad) ")
    plt.legend(loc="best")
    plt.grid(True)

    # 绘制速度 dq
    plt.figure(2)
    for i in range(1, 7):
        plt.plot(t, dq[i - 1, :], label="Joint " + str(i))
    plt.title("Joint Velocity (dq)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend(loc="best")
    plt.grid(True)

    # 绘制加速度 ddq
    plt.figure(3)
    for i in range(1, 7):
        plt.plot(t, ddq[i - 1, :], label="Joint " + str(i))
    plt.title("Joint Acceleration(ddq)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration(rad/s^2)")
    plt.legend(loc="best")
    plt.grid(True)

    # 写入文件夹
    q_left = np.asarray(q_left, dtype=float).reshape(-1)
    q_right = np.asarray(q_right, dtype=float).reshape(-1)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "Logs")
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, "bianshiguiji_%s.txt" % date.today().strftime("%Y%m%d"))
    with open(out_name, "w") as fid:
        for i in range(191):
            fid.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (
                q[0, i], q_left[i], q_right[i], q[3, i], q[4, i], q[5, i]))
    plt.show()
    return cond_aver


if __name__ == "__main__":
    main()
