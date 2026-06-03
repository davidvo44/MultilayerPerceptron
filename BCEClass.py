import numpy as np


class BinaryCrossEntropy(object):

    def compute(self, Y, A):
        return -np.mean(
            Y * np.log(A + 1e-8) +
            (1 - Y) * np.log(1 - A + 1e-8)
        )

    def derivative(self, Y, A):
        return A - Y