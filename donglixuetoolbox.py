import numpy as np
import matplotlib.pyplot as plt


def Link(*args, **kwargs):
    raise NotImplementedError("MATLAB Robotics Toolbox (RTB) function Link is not available")


def SerialLink(*args, **kwargs):
    raise NotImplementedError("MATLAB Robotics Toolbox (RTB) function SerialLink is not available")


def transl(*args, **kwargs):
    raise NotImplementedError("MATLAB Robotics Toolbox (RTB) function transl is not available")


def main():
    # 定义MDH参数 [alpha a d theta] (单位: rad/m)
    # 注意：theta在Robotics Toolbox中会自动加上关节变量q
    MDH = np.array([
        [0, 0, -0.269, 0],          # 关节1
        [np.pi / 2, 0, 0, np.pi / 2],  # 关节2 (d=0.138+0.024=0.162m)
        [np.pi / 2, 0, 0, np.pi],      # 关节3 (d=-0.127-0.024=-0.151m)
        [0, 0.360, 0, 0],              # 关节4 (a=360mm=0.36m)
        [0, 0.360, 0, 0],              # 关节5
        [np.pi / 2, 0, 0, 0],          # 关节6
        [0, 0.079, 0, 0],              # 末端工具
    ], dtype=float)

    # 创建Link对象（Modified DH）
    # 关节1
    L = [None] * 6
    L[0] = Link("d", MDH[0, 2], "a", MDH[0, 1], "alpha", MDH[0, 0], "offset", MDH[0, 3], "modified")
    # 关节1和关节2之间的平移关节
    L[1] = Link("d", MDH[1, 2], "a", MDH[1, 1], "alpha", MDH[1, 0], "offset", MDH[1, 3], "modified")
    L[2] = Link("d", MDH[2, 2], "a", MDH[2, 1], "alpha", MDH[2, 0], "offset", MDH[2, 3], "modified")
    L[3] = Link("d", MDH[3, 2], "a", MDH[3, 1], "alpha", MDH[3, 0], "offset", MDH[3, 3], "modified")
    L[4] = Link("d", MDH[4, 2], "a", MDH[4, 1], "alpha", MDH[4, 0], "offset", MDH[4, 3], "modified")
    L[5] = Link("d", MDH[5, 2], "a", MDH[5, 1], "alpha", MDH[5, 0], "offset", MDH[5, 3], "modified")

    # 设置动力学参数
    m = np.array([4.455, 0.87, 5.76, 1.33, 0.51, 1])
    r = [
        np.array([[0.0], [0.0], [-0.13]]),
        np.array([[0.0], [0.0], [0.0]]),
        np.array([[0.18], [0.0], [0.0]]),
        np.array([[0.18], [0.0], [0.0]]),
        np.array([[0.0], [0.0], [0.0]]),
        np.array([[0.05], [0.0], [0.0]]),
    ]
    I = [
        np.diag([0.023, 0.024, 0.016]),
        np.diag([0.004, 0.004, 0.004]),
        np.diag([0.134, 0.098, 0.046]),
        np.diag([0.027, 0.026, 0.0027]),
        np.diag([0.0006, 0.0006, 0.0007]),
        np.diag([0.0012, 0.0057, 0.009]),
    ]

    # 为每个Link对象赋动力学参数
    for i in range(6):
        L[i].m = m[i]
        L[i].r = r[i].T
        L[i].I = I[i]

    # 创建机械臂模型，现在是7自由度
    Six_Link = SerialLink(L, name="6-DOF Modified DH Arm", manufacturer="Custom")

    # 添加末端工具（最后一行MDH参数）
    Six_Link.tool = transl(MDH[6, 1], 0, MDH[6, 2])

    # 验证模型
    print("机械臂MDH参数:")
    print(Six_Link.mdh)  # 显示改进型DH参数

    # 运动学验证，现在需要7个关节角度
    q_test = np.array([0, 0.2, -0.3, 0.7, -0.5, 0])  # 新增的平移关节设为0
    T = Six_Link.fkine(q_test)
    print("末端位姿:")
    print(T)

    # 动力学计算
    q = np.array([0, 0.2, -0.3, 0.7, -0.5, 0])
    qd = np.ones(6)
    qdd = np.ones(6)
    gravity = np.array([0, 0, -9.8])
    F_tip = np.array([0, 0, 0, 0, 0, 0])

    # 计算关节力矩
    # Six_Link.rne is Robotics Toolbox (RTB). The original tries one argument
    # order and falls back to the other.
    try:
        tau = Six_Link.rne(q, qd, qdd, gravity, F_tip)
    except Exception:
        tau = Six_Link.rne(q, qd, qdd, F_tip, gravity)

    print("计算得到的关节力矩(Nm):")
    print(tau.T)

    # 可视化
    plt.figure(1)
    Six_Link.plot(q_test, jointdiam=1, view=[30, 30])
    plt.title("7-DOF机械臂模型(含关节间平移)")
    plt.show()
    return tau


if __name__ == "__main__":
    main()
