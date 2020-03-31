import numpy as np
from scipy.stats import norm

__author__ = 'Aryan Pedawi'


class UVDist:
    """univariate density class"""

    def __init__(self, pair_dict, tol=1e-10):
        assert len(pair_dict) >= 3
        assert all([isinstance(i, float) for i in pair_dict.keys()])
        assert all([isinstance(i, float) for i in pair_dict.values()])
        self.y = np.array(sorted(pair_dict.keys()))
        self.x = np.array([pair_dict[y] for y in self.y])
        self.h = np.diff(self.x)
        assert (self.y.min() == 0.) & (self.y.max() == 1.) & (self.h.min() > 0.)
        self.delta = np.diff(self.y) / self.h
        a = 1. / (self.h * self.delta)
        b = self.delta[:-1] / self.h[:-1] + self.delta[1:] / self.h[1:]
        c = 1. / self.h[:-1] + 1. / self.h[1:]
        d = np.zeros((2 + len(c),))
        condition = True
        while condition:
            _d = (c - a[:-1] * d[:-2] - a[1:] * d[2:]) ** 2
            _d += 4 * (a[:-1] + a[1:]) * b
            _d **= 0.5
            _d += c - a[:-1] * d[:-2] - a[1:] * d[2:]
            _d /= (2 * (a[:-1] + a[1:]))
            d[1:-1] = _d
            condition = np.abs(d[1:-1] - _d).max() > tol
        self.d = d

    def _cdf(self, x):
        idx = (self.x[:-1] <= x) & (self.x[1:] > x)
        if ~np.any(idx):
            if x >= self.x[-1]:
                return 1.
            else:
                return 0.
        theta = (x - self.x[:-1]) / self.h
        top = self.y[1:] * theta ** 2
        top += (self.y[1:] * self.d[:-1] + self.y[:-1] * self.d[1:]) / self.delta * theta * (1 - theta)
        top += self.y[:-1] * (1 - theta) ** 2
        bottom = theta ** 2
        bottom += (self.d[:-1] + self.d[1:]) / self.delta * theta * (1 - theta)
        bottom += (1 - theta) ** 2
        y = top / bottom
        return y[idx]

    def _icdf(self, y):
        assert (y >= 0) & (y <= 1)
        idx = (self.y[:-1] <= y) & (self.y[1:] > y)
        if ~np.any(idx):
            if y >= 1:
                return self.x[-1]
            else:
                return self.x[0]
        a = y * (2 - (self.d[:-1] + self.d[1:]) / self.delta)
        a -= self.y[:-1] + self.y[1:] - (self.y[1:] * self.d[:-1] + self.y[:-1] * self.d[1:]) / self.delta
        b = y * ((self.d[:-1] + self.d[1:]) / self.delta - 2)
        b -= (self.y[1:] * self.d[:-1] + self.y[:-1] * self.d[1:]) / self.delta - 2 * self.y[:-1]
        c = y - self.y[:-1]
        theta = -(b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        x = theta * self.h + self.x[:-1]
        return x[idx]

    def _pdf(self, x):
        idx = (self.x[:-1] <= x) & (self.x[1:] > x)
        if ~np.any(idx):
            if x >= self.x[-1]:
                return 0.
            else:
                return 0.
        theta = (x - self.x[:-1]) / self.h
        top = self.d[1:] * theta ** 2
        top += 2 * self.delta * theta * (1 - theta)
        top += self.d[:-1] * (1 - theta) ** 2
        bottom = theta ** 2
        bottom += (self.d[:-1] + self.d[1:]) / self.delta * theta * (1 - theta)
        bottom += (1 - theta) ** 2
        bottom **= 2
        dy = top / bottom
        return dy[idx]

    def cdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.vectorize(self._cdf)(x)

    def icdf(self, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.vectorize(self._icdf)(y)

    def pdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.vectorize(self._pdf)(x)

    def log_cdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(self.cdf(x))

    def log_pdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(self.pdf(x))

    def sample(self, n):
        assert isinstance(n, int) & (n > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.icdf(np.random.random((n,)))


class MVDist:
    """multivariate density class"""

    def __init__(self, pair_dicts, rho):
        rho = np.array(rho)
        assert isinstance(rho, np.ndarray) & (len(rho.shape) == 1)
        assert len(pair_dicts) == int(np.sqrt(2 * len(rho))) + 1
        self.nb_vars = len(pair_dicts)
        self.uv_densities = [UVDist(x) for x in pair_dicts]
        self.rho = rho

    def sample(self, n):
        assert isinstance(n, int) & (n > 0)
        out = np.random.multivariate_normal(mean=[0.] * self.nb_vars, cov=self.cov, size=n)
        out = norm().cdf(out)
        for i in range(self.nb_vars):
            out[:, i] = self.uv_densities[i].icdf(out[:, i])
        return out

    @property
    def cov(self):
        idx = np.tri(self.nb_vars, dtype=bool, k=-1)
        cov = np.zeros((self.nb_vars, self.nb_vars), dtype=float)
        cov[idx] = 2 * np.sin(np.array(self.rho) * np.pi / 6.)
        cov += cov.T + np.eye(self.nb_vars, dtype=float)
        return cov

    @property
    def _rho(self):
        idx = np.tril_indices(self.nb_vars, k=-1)
        return 6 / np.pi * np.arcsin(self.cov[idx] / 2.)