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


def menuLayer(dataset, choice):
    dataset = np.array(dataset)
    _, n = dataset.shape # m = ligne, n = colonne
    if choice ==  "Diagnostic Breast Cancer Data (project)":
        n -= 2
    else:
        n -= 1
    layerSize = click.prompt("\nNow, enter the number of Layer", type=int)
    time.sleep(1)
    layers = []
    layers.append(n);
    for i in range(1, layerSize - 1):
        neuron = click.prompt(f"\nNeuron from Layer {i + 1}", type=int)
        layers.append(neuron)
    if choice ==  "Diagnostic Breast Cancer Data (project)":
        n = 2
    else:
        n = 10
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
        f"Learning Rate : {parameter.learningRate}\n" \
        f"Batch Size : {parameter.batchSize}\n" \
        f"Epoch : {parameter.epoch}\n" \
        f"Loss Function: {parameter.loss}\n"  \
        f"EarlyStop: {parameter.earlyStop}\n"  \
        f"Optimiser: {parameter.optimiser}\n\n"  \
            "Select Parameter",
        
        choices=["Batch Size", "Epoch", "Loss", "Early Stop", "Optimiser", "Done"]
    ).execute()

def newParam(param):
    if param == "Format" or param == "Loss":
        newChange = click.prompt(f"\nNew data for {param}", type=str)
    else:
        newChange = click.prompt(f"\nNew data for {param}", type=int)
    return newChange


def addParameter():
    parameter = parameterClass.Parameter()
    while (1):
        choice = menuParameter(parameter)
        if choice == "Batch Size":
           parameter.batchSize = newParam("batch")
        elif choice == "Epoch":
           parameter.epoch = newParam("epoch")
        elif choice == "Loss":
           parameter.loss = newParamLoss()
        elif choice == "Optimiser":
           parameter.optimiser = newParamOpt()
        elif choice == "Early Stop":
           parameter.optimiser = newParamEarlyStop()
        elif choice == "Done":
            return parameter

def newParamLoss():
    return inquirer.select(
        message="\n\nChoose the Loss Algorithm",
        choices=["Standard","Categorical Crossentropy", "Binary Crossentropy"]
    ).execute()

def newParamOpt():
    return inquirer.select(
        message="\n\nChoose the Optimiser",
        choices=["Standard", "Adam"]
    ).execute()

def newParamEarlyStop():
    return inquirer.confirm(message="Activate Early Stop?").execute()