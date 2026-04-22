import click
import time
import networkClass
import graphClass
import parameterClass
import numpy as np


counter = 0
min_delta = 1e-3
best_loss = float("inf")
patience = 10


def train(trainCl, predictCl, NeuNetwork: networkClass, parameter: parameterClass):

    graph = graphClass.Graph()
    patience = 0
    counter = 0
    if trainCl == None or predictCl == None:
        click.echo(click.style("\nData is not separated\n   Return...", fg='red'))
        time.sleep(1)
        return;

    for epochI in range (parameter.epoch):
        indices = np.random.permutation(trainCl.data.shape[1])
        trainCl.data = trainCl.data[:, indices]
        trainCl.result = trainCl.result[indices]
        epoch_loss = 0
        num_batches = 0
        print(f"\nfor epoch {epochI}:")
        trainCl.resetEpoch()
        for i in range(0, trainCl.data.shape[1], parameter.batchSize):
            X_batch =  trainCl.data[:, i:i+parameter.batchSize]
            Y_batch = trainCl.result[i:i+parameter.batchSize]
            NeuNetwork.forwardPropagation(X_batch)
            dW, db = NeuNetwork.backwardPropagation(X_batch, Y_batch, parameter)
            if parameter.optimiser == "Standard":
                NeuNetwork.update(dW, db, parameter.learningRate)
            else:
                NeuNetwork.updateAdam(dW, db, epochI, parameter.learningRate)
            epoch_loss,num_batches = updateLoss(parameter,NeuNetwork, Y_batch, epoch_loss, num_batches)
        loss, accuracy = printEpochSituation(epoch_loss, num_batches, NeuNetwork, trainCl)
        graph.getPredictHisto(predictCl, NeuNetwork, parameter, loss, accuracy)
        parameter.learningRate *= 0.99
        if parameter.earlyStop == True:
            if earlyStop(loss, epochI) == True:
                break
    graph.drawCurve()

def earlyStop(loss, epochI):
    global best_loss, counter
    if best_loss - loss > 0:
        best_loss = loss
        counter = 0
    else:
        counter +=1
        print("repeat counter")
    if counter >= patience:
        print(f"Early stopping at epoch {epochI}")
        return True
    return False

def updateLoss(parameter,NeuNetwork, Y_batch, epoch_loss, num_batches):
    oneHot_Y = NeuNetwork.oneHot(Y_batch)
    if parameter.loss == "Categorical Crossentropy":
        batch_loss = -np.mean(np.sum(oneHot_Y * np.log(NeuNetwork.A[-1] + 1e-8), axis=0))
    else:
        batch_loss = -np.mean(np.sum(oneHot_Y * np.log(NeuNetwork.A[-1] + 1e-8) + (1 - oneHot_Y) * np.log(1 - NeuNetwork.A[-1] + 1e-8) , axis=0))
    epoch_loss += batch_loss
    num_batches += 1
    return epoch_loss, num_batches

def printEpochSituation(epoch_loss, num_batches, NeuNetwork, trainCl):
    loss = epoch_loss / num_batches
    print(f"Loss is {loss}")
    NeuNetwork.forwardPropagation(trainCl.data)
    pred = np.argmax(NeuNetwork.A[-1], axis=0)
    accuracy = np.mean(pred ==  trainCl.result)
    print(f"Accuracy: {accuracy}")
    return loss, accuracy