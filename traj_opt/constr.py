import numpy as np

from matlab_globals import colon


def constr(x):
    omega = 0.1 * np.pi
    t = colon(0, 1, 2 * np.pi / omega - 1)  # 采样点
    q = np.zeros((6, len(t)))
    dq = np.zeros((6, len(t)))
    ddq = np.zeros((6, len(t)))
    # disp('x');disp(x);
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
    # max(abs(q(1,:)))-15*pi/180>0 || max(abs(q(2,:)))-10*pi/180>0 || max(abs(q(3,:)+10*pi/180))-20*pi/180>0  ...
    #              ||max(abs(q(4,:)-30*pi/180))-30*pi/180>0||max(abs(q(5,:)))-15*pi/180>0||max(abs(q(6,:)))-15*pi/180>0
    f = np.zeros(8)
    f[0] = np.max(np.abs(q[0, :])) - 15 * np.pi / 180
    f[1] = np.max(np.abs(q[1, :])) - 10 * np.pi / 180
    f[2] = np.max(np.abs(q[2, :] + 10 * np.pi / 180)) - 20 * np.pi / 180
    f[3] = np.max(np.abs(q[3, :] - 25 * np.pi / 180)) - 25 * np.pi / 180
    f[4] = np.max(np.abs(q[4, :])) - 10 * np.pi / 180
    f[5] = np.max(np.abs(q[5, :])) - 10 * np.pi / 180
    # i is left at 6 by the for-loop above (MATLAB for-loop variable persists)
    f[6] = np.max(np.abs(dq[i - 1, :])) - np.pi * 0.25
    f[7] = np.max(np.abs(ddq[i - 1, :])) - np.pi * 0.25
    h = np.array([])
    return f, h
