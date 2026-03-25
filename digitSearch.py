import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('./train.csv')
data = np.array(data)
m, n = data.shape # m = ligne, n = colonne

def main():
    np.random.shuffle(data)

    data_dev = data[0:1000].T  
    Y_dev = data_dev[0]  #Extraire Feature, ne prend que la premieres lignes
    X_dev = data_dev[1:n]  #Prend tout sauf la premiere
    X_dev = X_dev / 255   #Normalisation#

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 225

    _,m_train = X_train.shape

    # print (softmax([2, 1, 0]));

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

"""Softmax Activation Function transforms a vector of numbers into a probability distribution where each value represents the likelihood of a particular class. 
    It is especially important for multi-class classification problems.

    - Each output value lies between 0 and 1.
    - The sum of all output values equals 1."""

def softmax(Z): 
    A = np.exp(Z) / sum(np.exp(Z))   # calcule exposant d'Euler avec les donnes fournies
    return A

"""Forward propagation in neural networks is the process where input data flows through each layer of the model to generate an output. 
It’s the step-by-step computation that transforms raw inputs into predictions using weights, biases and activation functions. 
This operation forms the backbone of how neural networks learn patterns and make decisions.

- It computes intermediate values layer by layer, starting from the input layer and ending at the output layer.
- Each neuron applies weighted sums and activation functions to extract features.
- It is used during both training and inference, but without weight updates.
- The accuracy of predictions heavily depends on how well forward propagation captures patterns from the input data."""

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # z = W * X + b     W: weight, x:input vector, b:bias
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0


"""One Hot Encoding,  encode variable under number"""

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


"""1. calcul erreur sortie      → dZ2
2. calcul gradient W2        → dW2
3. propager erreur arrière   → dZ1
4. calcul gradient W1        → dW1"""


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # Calcule difference entre prediction et verite
    dW2 = 1 / m * dZ2.dot(A1.T) # Calcul sur comment ajuster    W2: gradient = erreur * entree
    db2 = 1 / m * np.sum(dZ2) #le biais reçoit juste la somme des erreurs.
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # propage l’erreur vers l’arrière en appliquant la derivee de ReLU
    dW1 = 1 / m * dZ1.dot(X.T) # Meme logique que W2
    db1 = 1 / m * np.sum(dZ1) #Somme des erreurs
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # if i % 10 == 0:
        #     print("Iteration: ", i)
        #     predictions = get_predictions(A2)
        #     print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

if __name__ == "__main__":
    main()