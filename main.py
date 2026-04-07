import networkClass
import click
import time
import mainMenu
import pandas as pd
import separateData
import dataClass
import trainData
import predictData

def main():
    print("\nWelcome to Multilayer Perceptron project")
    time.sleep(1)
    print("Before the train start, we need to get some information")
    time.sleep(1)

    trainCl : dataClass.Data = None
    predictCl : dataClass.Data = None

    try:
        choiceDataset = mainMenu.menuData()
        if choiceDataset == "Diagnostic Breast Cancer Data (project)":
            dataset = "data.csv"
        else:
            dataset = "train.csv"
        data = pd.read_csv(dataset)
        layer = mainMenu.menuLayer(data, choiceDataset)
        neuNetwork = networkClass.Network(layer)
        parameter = mainMenu.addParameter()
        while (1):
            choice = mainMenu.programChoice()
            if choice == "Separate the dataset":
                predictCl, trainCl = separateData.SeparateData(data, choiceDataset)
            elif choice == "Train program":
                trainData.train(trainCl, predictCl, neuNetwork, parameter)
            elif choice == "Prediction program":
                predictData.predict(predictCl,neuNetwork)
            else:
                return
        # layerSize, layers = mainMenu.menuLayer()
    
    except KeyboardInterrupt:
        click.echo(click.style("\nForce Quit...", fg='red'))
        return
 

if __name__ == "__main__":
    main()
