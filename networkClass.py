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
    
    def ReLU_deriv(self, Z): #detection neuronne actif sous forme bool
        return Z > 0

    def softmax(self, Z): #softmax a stabiliser, utilite??
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    """One Hot Encoding,  encode variable under number"""

    def oneHot(self, Y):
        oneHot_Y = np.zeros((Y.size, Y.max() + 1))
        oneHot_Y[np.arange(Y.size), Y] = 1
        oneHot_Y = oneHot_Y.T
        return oneHot_Y


    """Need stock A in each layer to backpropagation"""
    def forwardPropagation(self, X: dataClass. Data):
        A = []
        Z = []
        A.append(X.data)

        for i in range (self.nb_layers - 1):
            tmpZ = np.dot(self.weights[i], A[i]) + self.biaises[i]
            Z.append(tmpZ)
            if i < (self.nb_layers - 2):
                A.append(self.ReLU(tmpZ))
            else:
                A.append(self.softmax(tmpZ))
        self.Z = Z
        self.A = A

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

    def backwardPropagation(self, X: dataClass.Data, learningRate):
        m = X.len
        L = self.nb_layers - 1
        dW = [0] * (L)
        db = [0] * (L)
        dZ = 0
        oneHot_Y = self.oneHot(X.result)
        
        compareValue = oneHot_Y
        for i in reversed(range(L)):
            A_prev = self.A[i]
            if (i == L - 1):
                dZ = self.A[L] - compareValue
            else:
                dA_prev = np.dot(self.weights[i].T, dZ)
                dZ = dA_prev * self.ReLU_deriv(self.Z[i - 1])
            dW[i] = (1 / m) * np.dot(dZ, A_prev.T)
            db[i] = (1 / m) * np.sum(dZ, axis=1, keepdims=True);
            print("tmpB:", db[i])
            print ("B:", self.biaises[i])
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



            # Z = self.A[self.nb_layers - 1 - i] - compareValue
            # tmpW = 1 / len(X.data) * Z.dot(self.nb_layers - 2 - i)
            # self.weights[self.nb_layers - 2 - i] -= learningRate * tmpW
            # tmpB = 1 / len(X.data) * np.sum(Z)
            # print (f"for {self.weights[self.nb_layers - 2 - i]}")


        # dZ2 = A - oneHot_Y # Calcule difference entre prediction et verite
        # dW2 = 1 / m * dZ2.dot(A1.T) # Calcul sur comment ajuster        W2: gradient = erreur * entree
        # db2 = 1 / m * np.sum(dZ2) #le biais reçoit juste la somme des erreurs.
        # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # propage l’erreur vers l’arrière en appliquant la derivee de ReLU
        # dW1 = 1 / m * dZ1.dot(X.T) # Meme logique que W2
        # db1 = 1 / m * np.sum(dZ1) #Somme des erreurs
        # return dW1, db1, dW2, db2



    def update(self, newW, newB):
        print("before", self.biaises[2])

        for i in range(len(self.biaises)):
            self.biaises[i] = newB[len(self.biaises) - 1 - i]
            self.weights[i] = newW[len(self.weights) - 1 - i]
        print("after", self.biaises[2])