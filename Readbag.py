import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter


def rosbag(filepath):
    raise NotImplementedError("MATLAB ROS Toolbox function rosbag is not available")


def select(bag, MessageType=None):
    raise NotImplementedError("MATLAB ROS Toolbox function select is not available")


def readMessages(joint_state):
    raise NotImplementedError("MATLAB ROS Toolbox function readMessages is not available")


def main():
    # fullfile('E:','study\ROSbag','result2.1.bag')
    filepath = os.path.join("E:", "study\\ROSbag", "result2.1.bag")
    bag = rosbag(filepath)
    joint_state = select(bag, MessageType="sensor_msgs/JointState")
    data = readMessages(joint_state)
    t_start = data[0].Header.Stamp.seconds
    num = len(data)
    print(num)
    ratio = 18.655 * 131072 / 6.8  # 应该是6.28
    p_start = np.zeros(7)
    for i in range(7):
        p_start[i] = data[0].Position[i]
        # p_start[i] = 0

    position = np.zeros((4534, 7))
    t = np.zeros(4534)
    for i in range(4534):
        position[i, 0] = data[i].Position[0] - p_start[0]  # QDD_1
        position[i, 1] = data[i].Position[1] - p_start[1]  # QDD_2
        position[i, 2] = (data[i].Position[2] - p_start[2]) / ratio  # Se_1
        position[i, 3] = (data[i].Position[3] - p_start[3]) / ratio  # Se_2
        position[i, 4] = (data[i].Position[4] - p_start[4])  # YS_L
        position[i, 5] = (data[i].Position[5] - p_start[5])  # YS_R
        position[i, 6] = (data[i].Position[6] - p_start[6])  # YS_M
        t[i] = data[i].Header.Stamp.seconds - t_start
    # idx_start = find(t >= 7.051, 1);%对应2.1
    # idx_end = find(t <= 27.051, 1, 'last');
    idx_start = int(np.flatnonzero(t >= 7.105)[0])  # 对应2.2
    idx_end = int(np.flatnonzero(t <= 27.105)[-1])
    data_of_interest = position[idx_start:idx_end + 1, :]
    time_of_interest = t[idx_start:idx_end + 1]
    plt.figure(1)
    # plt.plot(t, position*180/3.14);
    plt.plot(time_of_interest, data_of_interest * 180 / 3.14)
    plt.legend(["QDD_1", "QDD_2", "Se_1", "Se_2", "YS_L", "YS_R", "YS_M"])
    # 力矩读取
    torque = np.zeros((4534, 7))
    for i in range(4534):
        torque[i, 0] = data[i].Effort[0]
        torque[i, 1] = data[i].Effort[1]
        torque[i, 2] = data[i].Effort[2] * 0.48 / 1000
        torque[i, 3] = data[i].Effort[3] * 0.48 / 1000
        # torque[i, 2]=0; torque[i, 3]=0;
        torque[i, 4] = data[i].Effort[4] / 9.1
        torque[i, 5] = data[i].Effort[5] / 9.1
        torque[i, 6] = data[i].Effort[6] / 9.1
    # 低通滤波器
    fs = 100  # 采样频率
    fc_low = 1  # 低截止频率
    fc_high = 25  # 高截止频率
    N = 5  # 滤波器阶数
    b, a = butter(N, fc_low / (fs / 2), btype="low")
    torque_filter = lfilter(b, a, torque, axis=0)
    torque_of_interest = torque_filter[idx_start:idx_end + 1, :]
    plt.figure(2)
    plt.plot(time_of_interest, torque_of_interest * 180 / 3.14)
    plt.legend(["QDD_1", "QDD_2", "Se_1", "Se_2", "YS_L", "YS_R", "YS_M"])
    plt.show()
    del fc_high, filepath
    return data_of_interest, torque_of_interest


if __name__ == "__main__":
    main()
