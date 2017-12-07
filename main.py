import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from optim_algs import DirectDescent, IncrementalDescent, AdaptiveIncrementalDescent, \
    AdaptiveIncrementalDescentWithRandomization

m = 500
dim = 200
ys = [np.random.rand(dim) * 100 for _ in range(m)]


def sum_of_dists(x):
    return sum(max(norm(x - y, 2) - 1, 0) for y in ys)


def subgrad_helper_dists(x, y):
    if norm(x - y) <= 1:
        return np.zeros_like(x)
    else:
        diff = x - y
        z = y + (diff / norm(diff, 2))
        diff = x - z
        return diff / norm(diff, 2)


class SubGrad(object):
    def __init__(self, y, subgrad_helper):
        self.y = y
        self.subgrad_helper = subgrad_helper

    def calc(self, x):
        return self.subgrad_helper(x, self.y)


# subgrads = [(lambda x: subgrad_helper(x, y)) for y in ys]
subgrads = []
for y in ys:
    subgrads.append(SubGrad(y, subgrad_helper_dists))

if __name__ == '__main__':
    models = [
        (DirectDescent(), "vanilla descent with diminishing stepsize rule"),
        (IncrementalDescent(), "incremental descent with diminishing stepsize rule"),
        (AdaptiveIncrementalDescent(), "incremental descent with dynamic stepsize rule"),
        (AdaptiveIncrementalDescentWithRandomization(), "randomized incremental descent with dynamic stepsize rule"),
    ]

    for model, label in models:
        model.objective = sum_of_dists
        model.m = m
        model.subgrads = subgrads
        model.x = np.zeros(dim)

        evs, vals = model.optimize()
        plt.plot(evs, vals, label=label)

    plt.legend()
    plt.xlabel("evaluation times of subgradients")
    plt.ylabel("f(x)")
    plt.xlim(0, 10000)
    plt.show()
