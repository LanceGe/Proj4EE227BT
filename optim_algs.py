from random import choice

import numpy as np


class SubgradModel(object):
    def __init__(self):
        self.objective = None
        self.m = 0
        self.subgrads = []
        self.x = None

    def optimize(self, **kwargs):
        pass


class DirectDescent(SubgradModel):
    def __init__(self):
        super(DirectDescent).__init__()

    def optimize(self, **kwargs):
        subg = lambda x: sum(subgrad.calc(x) for subgrad in self.subgrads)
        k = 0
        a = lambda n: 1.0 / (1.0 + n)
        conv = 0
        ev = 0
        evs = []
        vals = []
        while conv < 10:
            val = self.objective(self.x)
            print("iteration %d: val = %f. " % (k, val))
            evs.append(ev)
            vals.append(val)
            self.x -= a(k) * subg(self.x)
            ev += self.m
            k += 1
            if np.isclose(val, self.objective(self.x)):
                conv += 1
            else:
                conv = 0
        return evs, vals


class IncrementalDescent(SubgradModel):
    def __init__(self):
        super(IncrementalDescent).__init__()

    def optimize(self, **kwargs):
        k = 0
        a = lambda n: 1.0 / (1 + n)
        conv = 0
        ev = 0
        evs = []
        vals = []
        while conv < 10:
            val = self.objective(self.x)
            print("iteration %d: val = %f. " % (k, val))
            for subg in self.subgrads:
                evs.append(ev)
                vals.append(self.objective(self.x))
                self.x -= a(k) * subg.calc(self.x)
                ev += 1
            k += 1
            if np.isclose(val, self.objective(self.x)):
                conv += 1
            else:
                conv = 0
        return evs, vals


class AdaptiveIncrementalDescent(SubgradModel):
    def __init__(self):
        super(AdaptiveIncrementalDescent).__init__()

    def optimize(self, **kwargs):
        k = 0
        conv = 0
        ev = 0
        evs = []
        vals = []
        sigma = 0.0
        delta = 1000.0
        B = 20.0
        l = 0
        K = dict()
        frec = dict()
        frec[-1] = float('inf')
        K[l] = 0
        tau = 1.0 / 2
        rho = 16
        beta = 1.0 / 2
        while conv < 10:
            val = self.objective(self.x)
            print("iteration %d: val = %f. " % (k, val))
            if val < frec[k - 1]:
                frec[k] = val
            else:
                frec[k] = frec[k - 1]

            if val <= frec[K[l]] - delta * tau:
                K[l + 1] = k
                sigma = 0.0
                delta *= rho
                l += 1
            elif sigma > B:
                K[l + 1] = k
                sigma = 0.0
                delta *= beta
                l += 1

            frec[k] = frec[K[l]] - delta
            a = 1.5 * (val - frec[k]) / (self.m ** 2)
            for subg in self.subgrads:
                evs.append(ev)
                vals.append(self.objective(self.x))
                self.x -= a * subg.calc(self.x)
                ev += 1
            k += 1
            if np.isclose(val, self.objective(self.x)):
                conv += 1
            else:
                conv = 0
        return evs, vals


class AdaptiveIncrementalDescentWithRandomization(SubgradModel):
    def __init__(self):
        super(AdaptiveIncrementalDescentWithRandomization).__init__()

    def optimize(self, **kwargs):
        p = 0
        conv = 0
        ev = 0
        evs = []
        vals = []
        sigma = 0.0
        delta = 1000.0
        B = 20.0
        l = 0
        P = dict()
        frec = dict()
        frec[-1] = float('inf')
        P[l] = 0
        tau = 1.0 / 2
        rho = 16
        beta = 1.0 / 2
        M = 100
        while conv < 10:
            val = self.objective(self.x)
            print("iteration %d: val = %f. " % (p, val))
            if val < frec[p - 1]:
                frec[p] = val
            else:
                frec[p] = frec[p - 1]

            if val <= frec[P[l]] - delta * tau:
                P[l + 1] = p
                sigma = 0.0
                delta *= rho
                l += 1
            elif sigma > B:
                P[l + 1] = p
                sigma = 0.0
                delta *= beta
                l += 1

            frec[p] = frec[P[l]] - delta
            a = 1.5 * (val - frec[p]) / (self.m * M)
            for _ in range(M):
                subg = choice(self.subgrads)
                evs.append(ev)
                vals.append(self.objective(self.x))
                self.x -= a * subg.calc(self.x)
                ev += 1
            p += 1
            if np.isclose(val, self.objective(self.x)):
                conv += 1
            else:
                conv = 0
            if p * M > 5000:
                break
        return evs, vals
