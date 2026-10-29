import numpy as np

from MDHTrans import MDHTrans


def main():
    # mdh坐标系测试 alpha a d theta
    MDH = np.array([
        [0, 0, -269, 0],
        [90, 0, 0, 90],
        [90, 0, 0, 180],
        [0, 360, 0, 0],
        [0, 360, 0, 0],
        [90, 0, 0, 0],
        [0, 79, 0, 0],
    ], dtype=float)  # 最后一行为到足底的距离
    q_deg = np.array([0, 0, -0.4, 1, -0.6, 0, 0], dtype=float)
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

    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67
    return T


if __name__ == "__main__":
    main()
