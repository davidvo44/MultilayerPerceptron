import click
import time
import networkClass
import parameterClass

def train(trainCl, predictCl, NeuNetwork: networkClass, parameter: parameterClass):
    if trainCl == None or predictCl == None:
        click.echo(click.style("\nData is not separated\n   Return...", fg='red'))
        time.sleep(1)
        return;
    NeuNetwork.forwardPropagation(trainCl)
    newW, newB = NeuNetwork.backwardPropagation(trainCl, 0.01)
    # NeuNetwork.update(newW, newB)
    



# def gradient_descent(X, Y, alpha, iterations):
# W1, b1, W2, b2 = init_params()
# for i in range(iterations):
#     Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
#     dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
#     W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#     # if i % 10 == 0:
#     #     print("Iteration: ", i)
#     #     predictions = get_predictions(A2)
#     #     print(get_accuracy(predictions, Y))
# return W1, b1, W2, b2