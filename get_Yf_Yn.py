import numpy as np

from getS import getS


def get_Yf_Yn(id, R, H, A, Yf_next, Yn_next, Po):
    c1 = (id - 1) * 10  # MATLAB c1=(id-1)*10+1, inclusive slice
    c2 = (id - 1) * 10 + 10
    Yf = np.zeros((3, 60))
    Yf[:, c1:c2] = H
    Yf = Yf + R @ Yf_next
    Yn = np.zeros((3, 60))
    Yn[:, c1:c2] = A
    Yn = Yn + R @ Yn_next + getS(Po) @ R @ Yf_next
    # disp('YFYN');disp(id);
    return Yf, Yn
