import pandas as pd
import numpy as np
import dataClass


def SeparateData(data, choice):
    if choice == "Diagnostic Breast Cancer Data (project)":
        return SeparateDataWDBC(data)
    else :
        return SeparateDataMNIST(data)

def SeparateDataMNIST(data: pd):
    data = data.sample(frac=1).reset_index(drop=True)
    data = np.array(data)
    lenData, lenFeat = data.shape


    data_predic = data[0:int(lenData / 2)].T
    Y_predic = data_predic[0]
    X_predic = data_predic[1:lenFeat]
    X_predic = X_predic / 255
    X_predic = X_predic - 0.5
    # """normalisation /255 si data : MNIST"""


    print(lenData)
    data_train = data[int(lenData / 2): lenData].T
    Y_train = data_train[0]
    X_train = data_train[1:lenFeat]
    X_train = X_train / 255
    X_train = X_train - 0.5
    _,m_train = X_train.shape


    predictCl = dataClass.Data(lenData / 2, X_predic, Y_predic)
    
    trainCl = dataClass.Data(m_train, X_train, Y_train)


    return predictCl, trainCl


    
def SeparateDataWDBC(data: pd):
    data = data.sample(frac=1).reset_index(drop=True)
    data = np.array(data)
    lenData, lenFeat = data.shape
    mapping = {'M': 1, 'B': 0}


    data_predict = data[0:int(lenData / 2)].T
    iD_predict = data_predict[0]
    labels = data_predict[1]
    Y_predict = np.vectorize(mapping.get)(labels)
    X_predict = data_predict[2:lenFeat]
    X_predict = normalize(X_predict, lenFeat)
    X_predict = X_predict.astype(float)
    X_predict -= 0.5 
    X_predict = np.round(X_predict, 4)

    data_train = data[int(lenData / 2): lenData].T
    iD_train = data_train[0]
    labels = data_train[1]
    Y_train = np.vectorize(mapping.get)(labels)
    X_train = data_train[2:lenFeat]
    X_train = normalize(X_train, lenFeat)
    X_train = X_train.astype(float)
    X_train -= 0.5 
    X_train = np.round(X_train, 4)
    _,m_train = X_train.shape


    predictCl = dataClass.Data(lenData / 2, X_predict, Y_predict)
    
    trainCl = dataClass.Data(m_train, X_train, Y_train)


    return predictCl, trainCl

def normalize(data, lenFeat):
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    denomi = max_vals - min_vals
    denomi[denomi == 0] = 1
    return (data - min_vals) / denomi