from InquirerPy import inquirer
import time
import click
import numpy as np
import parameterClass

def menuData():
    return inquirer.select(
        message="\n\nChoose the Dataset",
        choices=["Diagnostic Breast Cancer Data (project)", "MNIST Data (tuto)"]
    ).execute()


def programChoice():
    return inquirer.select(
        message="\n\nMain Menu",
        choices=["Separate the dataset", "Train program",
                 "Prediction program", "Quit"]
    ).execute()


def menuLayer(dataset):
    dataset = np.array(dataset)
    _, n = dataset.shape # m = ligne, n = colonne
    n -= 1; #input
    layerSize = click.prompt("\nNow, enter the number of Layer", type=int)
    time.sleep(1)
    layers = []
    layers.append(n);
    for i in range(1, layerSize - 1):
        neuron = click.prompt(f"\nNeuron from Layer {i + 1}", type=int)
        layers.append(neuron)
    n = 10; #output
    layers.append(n);
    return layers

def programChoice():
    return inquirer.select(
        message="\n\nMain Menu",
        choices=["Separate the dataset", "Train program",
                 "Prediction program", "Quit"]
    ).execute()

def menuParameter(parameter : parameterClass):
    return inquirer.select(
        message="\n\nparameter Information:\n" \
        f"Learning Rate Format: {parameter.learningRateFormat}\n" \
        f"Learning Rate : {parameter.learningRate}\n" \
        f"Batch Size : {parameter.batchSize}\n" \
        f"Loss Function: {parameter.loss}\n\n"  \
            "Select Parameter",
        
        choices=["Learning Rate Format", "Batch Size", "Epoch", "Loss", "Done"]
    ).execute()

def addParameter():
    parameter = parameterClass.Parameter()
    while (1):
        choice = menuParameter(parameter)
        if choice == "Learning Rate Format":
            print("prediction!!!")
        elif choice == "Batch Size":
            print("prediction!!!")
        elif choice == "Epoch":
            print("prediction!!!")
        elif choice == "Loss":
            print("prediction!!!")
        elif choice == "Done":
            return parameter
