
import numpy as np


def calibrate_threshold(u_val, q=95.0):
    return float(np.percentile(np.asarray(u_val, dtype=np.float64), q))


def accept(u, tau):
    return np.asarray(u) <= tau
