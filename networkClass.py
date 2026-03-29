import numpy as np
import dataClass

class Network(object):

    def __init__(self, sizes):
        self.nb_layers = len(sizes)
        self.sizes = sizes
        self.biaises = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
        self.Z = []
        self.A = []

    def gradient_descent(epoch):
        for i in range(epoch):
            print("forward_prop")
            print("backward_prop")
            print("update_params")
            print("Epoch {i}")


    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def ReLU(self, Z):
        return np.maximum(Z, 0)
    
    def LeakyReLU(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def ReLU_deriv(self, x): #detection neuronne actif sous forme bool
        return np.where(x > 0, 1, 0.01)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=0, keepdims=True)  # stabilité numérique
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    """One Hot Encoding,  encode variable under number"""

    def oneHot(self, Y):
        oneHot_Y = np.zeros((Y.size,10))
        oneHot_Y[np.arange(Y.size), Y] = 1
        oneHot_Y = oneHot_Y.T
        return oneHot_Y


    """Need stock A in each layer to backpropagation"""
    def forwardPropagation(self, A):
        self.A = []
        self.Z = []

        self.A.append(A)

        for i in range(self.nb_layers - 1):
            Z = np.dot(self.weights[i], A) + self.biaises[i]

            if i < self.nb_layers - 2:
                A = self.LeakyReLU(Z)
            else:
                A = self.softmax(Z)

            self.A.append(A)
            self.Z.append(Z)

    """ Si 4 layer:
            A[0] = data input
            A[1] = Premiere Transfo avec ReLU situe en couche Hidden1
            A[2] =  Transfo avec ReLU situe en couche Hidden2
            A[3] = Resultat Final Utilise avec SoftMax

            Z[0] = Resultat Regression Lineaire avec Data * W[0] + B[1] -> Hidden Layer 1
            Z[1] = Resultat Regression Lineaire avec A[1] * W[1] + B[1] -> Hidden Layer 2
            Z[2] = Resultat Regression Lineaire avec A[2] * W[2] + B[2] -> Final Layer

        Weight = 3;
        Biais = 3;
        A = 4 dont data init en A[0]
        Z = 3

    """

    def backwardPropagation(self, X, Y, learningRate):
        m = X.shape[1]
        L = self.nb_layers - 1
        dW = [0] * (L)
        db = [0] * (L)
        dZ = 0
        oneHot_Y = self.oneHot(Y)
        for i in reversed(range(L)):
            A_prev = self.A[i]
            if (i == L - 1):
                dZ = self.A[L] - oneHot_Y
            else:
                dA_prev = np.dot(self.weights[i + 1].T, dZ)
                dZ = dA_prev * self.ReLU_deriv(self.Z[i])
            dW[i] = (1 / m) * np.dot(dZ, A_prev.T)
            db[i] = (1 / m) * np.sum(dZ, axis=1, keepdims=True);
        loss = -np.mean(np.sum(oneHot_Y * np.log(self.A[-1] + 1e-8), axis=0))
        return dW, db
    
    # m = X.len
    # Y = X.result
    # oneHot_Y = self.oneHot(Y)

    # dW = [0] * (self.nb_layers - 1)
    # db = [0] * (self.nb_layers - 1)

    # dZ = self.A[-1] - oneHot_Y

    # for i in reversed(range(self.nb_layers - 1)):

    #     A_prev = self.A[i]

    #     # gradients
    #     dW[i] = (1 / m) * np.dot(dZ, A_prev.T)
    #     db[i] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    #     # backprop vers couche précédente
    #     if i > 0:
    #         dA_prev = np.dot(self.weights[i].T, dZ)
    #         dZ = dA_prev * self.ReLU_deriv(self.Z[i - 1])

    # return dW, db

    # one_hot_Y = one_hot(Y)
    # dZ2 = A2 - one_hot_Y
    # dW2 = 1 / m * dZ2.dot(A1.T)
    # db2 = 1 / m * np.sum(dZ2)
    # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    # dW1 = 1 / m * dZ1.dot(X.T)
    # db1 = 1 / m * np.sum(dZ1)
    # return dW1, db1, dW2, db2



    def update(self, dW, db, learningRate):

        for i in range(len(self.biaises)):
            self.biaises[i] -= learningRate * db[i]
            self.weights[i] -= learningRate * dW[i]

