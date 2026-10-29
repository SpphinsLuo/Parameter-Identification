import numpy as np
import matplotlib.pyplot as plt

from Leftleg import Leftleg
from Rightleg import Rightleg
from matlab_globals import colon


def main():
    # x is a MATLAB base-workspace variable (Fourier coefficients). It is not
    # defined in this script.
    omega = 0.01 * np.pi
    t = colon(0, 0.01, 2 * np.pi / omega)  # 采样点
    q = np.zeros((6, len(t)))
    dq = np.zeros((6, len(t)))
    ddq = np.zeros((6, len(t)))

    # disp('x');disp(x);
    x_coeff = np.asarray(x, dtype=float).reshape(-1, order="F")
    for i in range(1, 7):
        p = x_coeff[(i * 7 - 6) - 1:i * 7]
        q[i - 1, :] = (p[0] + p[1] * np.sin(omega * t) + p[2] * np.cos(omega * t)
                       + p[3] * np.sin(2 * omega * t) + p[4] * np.cos(2 * omega * t)
                       + p[5] * np.sin(3 * omega * t) + p[6] * np.cos(3 * omega * t))
        dq[i - 1, :] = (p[1] * omega * np.cos(omega * t) - p[2] * omega * np.sin(omega * t)
                        + p[3] * 2 * omega * np.cos(2 * omega * t) - p[4] * 2 * omega * np.sin(2 * omega * t)
                        + p[5] * 3 * omega * np.cos(3 * omega * t) - p[6] * 3 * omega * np.sin(3 * omega * t))
        ddq[i - 1, :] = (-p[1] * omega ** 2 * np.sin(omega * t) - p[2] * omega ** 2 * np.cos(omega * t)
                         - p[3] * 4 * omega ** 2 * np.sin(2 * omega * t) - p[4] * 4 * omega ** 2 * np.cos(2 * omega * t)
                         - p[5] * 9 * omega ** 2 * np.sin(3 * omega * t) - p[6] * 9 * omega ** 2 * np.cos(3 * omega * t))

    Leftside = Leftleg(q[1, :], q[2, :])
    Rightside = Rightleg(q[1, :], q[2, :])
    # plot(t, q(1:6,:)) plots each row of q against t
    plt.plot(t, q.T)
    plt.xlabel("时间 t (s)")
    plt.ylabel("关节角度 q")
    plt.title("所有关节角度随时间变化")
    plt.legend(["关节1", "关节2", "关节3", "关节4", "关节5", "关节6"])
    # 保存数据
    np.savetxt("bianshiguanjiejiao_1120.csv", q.T, delimiter=",", fmt="%.5g")
    plt.show()
    return Leftside, Rightside, dq, ddq


if __name__ == "__main__":
    main()
