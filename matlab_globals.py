"""Shared MATLAB ``global`` workspace for converted modules.

This file is not a converted source file. MATLAB ``global`` names are visible
to every function that declares them; ``G`` is the Python equivalent.
"""
import numpy as np

G = {}


def colon(start, step, stop):
    """MATLAB colon expression ``start:step:stop``.

    The stop value is included when it lies on the grid (within floating-point
    tolerance), matching MATLAB.
    """
    start = float(start)
    step = float(step)
    stop = float(stop)
    if step == 0.0:
        raise ValueError("colon step must be nonzero")
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    if n <= 0:
        return np.array([], dtype=float)
    return start + step * np.arange(n, dtype=float)
