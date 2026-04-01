import click
import time
import networkClass
import parameterClass
import numpy as np

def train(trainCl, predictCl, NeuNetwork: networkClass, parameter: parameterClass):
    epoch_loss = 0
    num_batches = 0
    best_loss = float("inf")
    patience = 10
    counter = 0
    min_delta = 1e-3

    if trainCl == None or predictCl == None:
        click.echo(click.style("\nData is not separated\n   Return...", fg='red'))
        time.sleep(1)
        return;

    with open("houses.csv", 'a') as f:
        f.write(f"Weight Init:\n{NeuNetwork.weights}")
        f.write(f"\n biais:\n{NeuNetwork.biaises}")
    
    for epochI in range (parameter.epoch):
        print(f"\nfor epoch {epochI}:")
        trainCl.resetEpoch()
        for i in range(0, trainCl.data.shape[1], parameter.batchSize):
            X_batch =  trainCl.data[:, i:i+parameter.batchSize]
            Y_batch = trainCl.result[i:i+parameter.batchSize]
            NeuNetwork.forwardPropagation(X_batch)
            dW, db = NeuNetwork.backwardPropagation(X_batch, Y_batch)
            NeuNetwork.update(dW, db, parameter.learningRate)
            oneHot_Y = NeuNetwork.oneHot(Y_batch)
            batch_loss = -np.mean(np.sum(oneHot_Y * np.log(NeuNetwork.A[-1] + 1e-8), axis=0))
            epoch_loss += batch_loss
            num_batches += 1
        loss = epoch_loss / num_batches
        print(f"Loss is {loss}")
        if parameter.earlyStop == True:
            if best_loss - loss > min_delta:
                best_loss = loss
                counter = 0
            else:
                counter +=1
            if counter >= patience:
                print(f"Early stopping at epoch {epochI}")
                break
        NeuNetwork.forwardPropagation(trainCl.data)
        pred = np.argmax(NeuNetwork.A[-1], axis=0)
        accuracy = np.mean(pred ==  trainCl.result)
        print(f"Accuracy: {accuracy}")
        parameter.learningRate *= 0.99
        with open("houses.csv", 'a') as f:
            f.write(f"\n\n\n\nfor epoch {epochI}:\n Weight:\n{NeuNetwork.weights}")
            f.write(f"\n biais:\n{NeuNetwork.biaises}")
    # test = [[1,2,3], [2,3,8]]
    # NeuNetwork.softmax(test)

