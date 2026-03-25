import numpy as np
import dataClass

class Network(object):

    def __init__(self, sizes):
        self.nb_layers = len(sizes)
        self.sizes = sizes
        self.biaises = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
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
    
    def softmax(self, Z): 
        A = np.exp(Z) / sum(np.exp(Z))   # calcule exposant d'Euler avec les donnes fournies
        return A
    
    """One Hot Encoding,  encode variable under number"""

    def oneHot(self, Y):
        oneHot_Y = np.zeros((Y.size, Y.max() + 1))
        oneHot_Y[np.arange(Y.size), Y] = 1
        oneHot_Y = oneHot_Y.T
        return oneHot_Y


    """Need stock A in each layer to backpropagation"""
    def forwardPropagation(self, X: dataClass. Data):
        A = [];
        Z = [];
        A.append(X.data)

        for i in range (self.nb_layers - 1):
            Z.append(np.dot(self.weights[i], A[i]) + self.biaises[i])
            if i < (self.nb_layers - 1):
                A.append(self.ReLU(Z))
            else:
                A.append(self.softmax(Z))
        self.Z = Z
        self.A = A


    def backwardPropagation(self, X: dataClass.Data, A, learningRate):

        Y = X.result
        print(self.nb_layers)
        oneHot_Y = self.oneHot(Y)
        for i in range(self.nb_layers - 1):
            print(self.nb_layers - 2 - i)
            Z = A[self.nb_layers - 1 - i] - oneHot_Y
            tmpW = 1 / len(X.data) * Z.dot(self.nb_layers - 2 - i)
            self.weights[self.nb_layers - 2 - i] -= learningRate * tmpW
            # tmpB = 1 / len(X.data) * np.sum(Z)
            print (f"for {self.weights[self.nb_layers - 2 - i]}")


        # dZ2 = A - oneHot_Y # Calcule difference entre prediction et verite
        # dW2 = 1 / m * dZ2.dot(A1.T) # Calcul sur comment ajuster        W2: gradient = erreur * entree
        # db2 = 1 / m * np.sum(dZ2) #le biais reçoit juste la somme des erreurs.
        # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # propage l’erreur vers l’arrière en appliquant la derivee de ReLU
        # dW1 = 1 / m * dZ1.dot(X.T) # Meme logique que W2
        # db1 = 1 / m * np.sum(dZ1) #Somme des erreurs
        # return dW1, db1, dW2, db2


"""1. calcul erreur sortie      → dZ2
2. calcul gradient W2        → dW2
3. propager erreur arrière   → dZ1
4. calcul gradient W1        → dW1"""


    # Z1 = W1.dot(X) + b1       # z = W * X + b     W: weight, x:input vector, b:bias
    # A1 = ReLU(Z1)
    # Z2 = W2.dot(A1) + b2
    # A2 = softmax(Z2)
    # return Z1, A1, Z2, A2
