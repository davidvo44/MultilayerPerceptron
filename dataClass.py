import numpy as np

class Data(object):

    def __init__(self, len, data, result):
        self.len = len
        self.data = data
        self.result = result
    
    def data(self):
        return self.data
    
    def result(self):
        return self.result
    
    def resetEpoch(self):
        perm = np.random.permutation(self.data.shape[1])  # génère un ordre aléatoire
        X_data_shuffled = self.data[:, perm]
        Y_data_shuffled = self.result[perm]
        self.data = X_data_shuffled
        self.result = Y_data_shuffled

