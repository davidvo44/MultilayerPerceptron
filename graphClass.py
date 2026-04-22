import numpy as np
import matplotlib.pyplot as plt
import networkClass

class Graph(object):

    def __init__(self):
        self.lossTrain = []
        self.lossPredict = []
        self.accTrain = []
        self.accPredict = []
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value


    def __getattr__(self, item):
        return (self.__dict__[item])
    
    def getPredictHisto(self, predictCl, NeuNetwork, parameter, loss, accuracy):
        predictNN = NeuNetwork.copy()
        predictNN.forwardPropagation(predictCl.data)
        oneHot_Y = predictNN.oneHot(predictCl.result)
        if parameter.loss == "Categorical Crossentropy":
            lossPredict = -np.mean(np.sum(oneHot_Y * np.log(predictNN.A[-1] + 1e-8), axis=0))
        else:
            lossPredict = -np.mean(np.sum(oneHot_Y * np.log(predictNN.A[-1] + 1e-8) + (1 - oneHot_Y) * np.log(1 - predictNN.A[-1] + 1e-8) , axis=0))
        pred = np.argmax(predictNN.A[-1], axis=0)
        accuracyPredict = np.mean(pred ==  predictCl.result)
        self.lossTrain.append(loss)
        self.lossPredict.append(lossPredict)
        self.accTrain.append(accuracy)
        self.accPredict.append(accuracyPredict)


    def drawCurve(self):

        self.drawLoss()
        self.drawAccuracy()


    def drawLoss(self):
        plt.plot(self.lossTrain, label="Training Loss")
        plt.plot(self.lossPredict, label="Validation Loss")
        plt.title("Loss accracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("lossFigure.png")
        plt.close();

    def drawAccuracy(self):
        plt.plot(self.accTrain, label="Training accuracy")
        plt.plot(self.accPredict, label="Validation accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("acuracy")
        plt.legend()
        plt.savefig("AccuracyFigure.png")
        plt.close();