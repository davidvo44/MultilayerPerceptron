import numpy as np


class CategoricalCrossEntropy:

    def compute(self, Y, A):

        return -np.mean(
            np.sum(
                Y * np.log(A + 1e-8),
                axis=0
            )
        )

    def derivative(self, Y, A):

        return A - Y