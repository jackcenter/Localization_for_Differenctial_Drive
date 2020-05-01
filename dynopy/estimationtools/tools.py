import numpy as np
from scipy import linalg


def monte_carlo_sample(mu: np.ndarray, r: np.ndarray, t=1):
    """
    creates a monte carlo sample from the mean and covariance
    :param mu: distribution mean
    :param r: distribution covariance
    :param t: number of samples to generate
    :return:
    """
    p = np.size(r, 0)

    s_v = linalg.cholesky(r, lower=True)
    q_k = np.random.randn(p, t)
    sample = mu + s_v @ q_k

    return sample


class DiscreteLinearStateSpace:
    def __init__(self, f, g, h, m, q, r, dt):
        self.F = f
        self.G = g
        self.H = h
        self.M = m
        self.Q = q
        self.R = r
        self.dt = dt

    def get_dimensions(self):
        return [np.size(self.F, 1), np.size(self.H, 1)]