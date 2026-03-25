import pandas as pd
import numpy as np
import dataClass

def SeparateData(data: pd):
    data = data.sample(frac=1).reset_index(drop=True)
    data = np.array(data)
    lenData, lenFeat = data.shape


    data_predic = data[0:int(lenData / 2)].T
    Y_predic = data_predic[0]
    X_predic = data_predic[1:lenFeat]
    X_predic = X_predic / 255
    # """normalisation /255 si data : MNIST"""


    print(lenData)
    data_train = data[int(lenData / 2): lenData].T
    Y_train = data_train[0]
    X_train = data_train[1:lenFeat]
    X_train = X_train / 255
    _,m_train = X_train.shape


    predictCl = dataClass.Data(lenData / 2, X_predic, Y_predic)
    trainCl = dataClass.Data(m_train, X_train, Y_train)


    return predictCl, trainCl


    """normalisation /255 si data : MNIST"""
    
