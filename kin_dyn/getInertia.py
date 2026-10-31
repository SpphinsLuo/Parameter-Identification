import numpy as np


def getInertia(xxi_, yyi_, zzi_, xyi_, xzi_, yzi_, xi_, yi_, zi_, ai, di, alphai, YYi, MZi, Mi, mi_):
    XY_ = xyi_ + ai * np.sin(alphai) * (MZi + di * Mi)
    XZ_ = xzi_ - ai * np.cos(alphai) * (MZi + di * Mi)
    YY_ = yyi_ + np.cos(alphai) ** 2 * YYi + 2 * di * np.cos(alphai) ** 2 * MZi + (di ** 2 * np.cos(alphai) ** 2 + ai ** 2) * Mi
    YZ_ = yzi_ + np.cos(alphai) * np.sin(alphai) * (YYi + 2 * di * MZi + di ** 2 * Mi)
    ZZ_ = zzi_ + np.sin(alphai) ** 2 * YYi + 2 * di * np.sin(alphai) ** 2 * MZi + (di ** 2 * np.sin(alphai) ** 2 + ai ** 2) * Mi
    XX_ = xxi_ + YYi + 2 * di * MZi + di ** 2 * Mi - YY_
    MX_ = mi_ * xi_ + ai * Mi
    MY_ = mi_ * yi_ - np.sin(alphai) * (MZi + di * Mi)
    MZ_ = mi_ * zi_ + np.cos(alphai) * (MZi + di * Mi)
    M_ = mi_ + Mi
    return XX_, XY_, XZ_, YY_, YZ_, ZZ_, MX_, MY_, MZ_, M_
