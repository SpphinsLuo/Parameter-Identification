import numpy as np

# 齐次变换矩阵的计算函数，使用MDH参数
def MDHTrans(alpha, a, d, theta):
    # alpha, a, d, theta 是 MDH 参数（alpha 和 theta 可以是弧度或角度）
    alpha = np.deg2rad(alpha)  # 如果输入是角度，转换为弧度
    theta = np.deg2rad(theta)
    
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    return T
