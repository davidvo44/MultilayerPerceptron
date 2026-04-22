import numpy as np
import dataClass
import copy


class Network(object):

    def __init__(self, sizes):
        self.nb_layers = len(sizes)
        self.sizes = sizes
        self.biaises = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
        self.Z = []
        self.A = []

        self.mW = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biaises]
        self.vW = [np.zeros_like(w) for w in self.weights]
        self.vb = [np.zeros_like(b) for b in self.biaises]

    def gradient_descent(epoch):
        for i in range(epoch):
            print("forward_prop")
            print("backward_prop")
            print("update_params")
            print("Epoch {i}")
    
    def copy(self):
        return copy.deepcopy(self)

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def ReLU(self, Z):
        return np.maximum(Z, 0)
    
    def LeakyReLU_deriv(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def ReLU_deriv(self, x): #detection neuronne actif sous forme bool
        return np.where(x > 0, 1, 0.01)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=0, keepdims=True)  # stabilité numérique
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    """One Hot Encoding,  encode variable under number"""

    def oneHot(self, Y):
        oneHot_Y = np.zeros((Y.size,self.sizes[-1]))
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

    def backwardPropagation(self, X, Y, parameter):
        m = X.shape[1]
        L = self.nb_layers - 1
        dW = [0] * (L)
        db = [0] * (L)
        dZ = 0
        oneHot_Y = self.oneHot(Y)
        for i in reversed(range(L)):
            A_prev = self.A[i]
            if (i == L - 1):
                if parameter.loss == "Standard":
                    dZ = self.A[L] - oneHot_Y #Softmax and Cross Entropy
                if parameter.loss == "Categorical Crossentropy":
                    loss = -np.mean(np.sum(oneHot_Y * np.log(self.A[L] + 1e-8), axis=0))
                    dA = - (oneHot_Y / (self.A[L] + 1e-8))
                if parameter.loss == "Binary Crossentropy":
                    dZ = self.A[L] - oneHot_Y
            else:
                dA_prev = np.dot(self.weights[i + 1].T, dZ)
                dZ = dA_prev * self.ReLU_deriv(self.Z[i])
            dW[i] = (1 / m) * np.dot(dZ, A_prev.T)
            db[i] = (1 / m) * np.sum(dZ, axis=1, keepdims=True);
        return dW, db
    
    """
    Cross Entropy: 
        - oneHot_Y * np.log(self.A[-1] + 1e-8) -> multiplication élément par élément
        oneHot_Y contient surtout des 0 sauf 1 à la position de la vraie classe
          Donc ça revient à garder uniquement : log(proba de la bonne classe)
        exemples:
        oneHot_Y      = [0, 1, 0]
        prediction    = [0.1, 0.7, 0.2]
            → résultat = [0, log(0.7), 0]
    """


    def update(self, dW, db, learningRate):

        for i in range(len(self.biaises)):
            self.biaises[i] -= learningRate * db[i]
            self.weights[i] -= learningRate * dW[i]


    def updateAdam(self, dW, db, t, learningRate):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = t + 1
        for i in range(len(self.weights)):
            # Update biased first moment estimate.
            # m is the exponentially moving average of the gradients.
            # beta1 is the decay rate for the first moment.
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]

            # Update biased second raw moment estimate.
            # v is the exponentially moving average of the squared gradients.
            # beta2 is the decay rate for the second moment.
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)

            # Compute bias-corrected first moment estimate.
            # This corrects the bias in the first moment caused by initialization at origin.
            mW_hat = self.mW[i] / (1 - beta1 ** t)
            mb_hat = self.mb[i] / (1 - beta1 ** t)

            vW_hat = self.vW[i] / (1 - beta2 ** t)
            vb_hat = self.vb[i] / (1 - beta2 ** t)

            #Upgrade
            self.biaises[i] -= learningRate * mb_hat / (np.sqrt(vb_hat) + epsilon)
            self.weights[i] -= learningRate * mW_hat / (np.sqrt(vW_hat) + epsilon)