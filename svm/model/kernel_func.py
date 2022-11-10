import numpy as np
import math


class Linear_Kernel:

    def run(self, x, y):
        return sum(x * y)


class Poly_Kernel:
    def __init__(self, d):
        self.d = d

    def run(self, x, y):
        return math.pow(sum(x * y), self.d)


class Gauss_Kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def run(self, x, y):
        return math.exp(-sum((x - y) * (x - y)) / (2 * self.sigma * self.sigma))


class Laplace_Kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def run(self, x, y):
        return math.exp(-math.sqrt(sum((x - y) * (x - y))) / self.sigma)


class Sigmoid_Kernel:
    def __init__(self, beta, zeta):
        self.beta = beta
        self.zeta = zeta

    def run(self, x, y):
        return math.tanh(self.beta * sum(x * y) + self.zeta)
