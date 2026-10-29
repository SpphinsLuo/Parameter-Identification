import numpy as np


def motion_para_clc(R, dq, ddq, wi, dwi, dvi, Pi, Pci, axisi):
    w = R @ wi + dq * axisi
    dw = R @ dwi + np.cross(R @ wi, dq * axisi) + ddq * axisi
    dv = R @ (np.cross(wi, Pi) + np.cross(wi, np.cross(wi, Pi)) + dvi)
    dvc = np.cross(dw, Pci) + np.cross(w, np.cross(w, Pci)) + dv
    return w, dw, dv, dvc
