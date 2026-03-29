import networkClass
import numpy as np
def predict(predictCl, NeuNetwork: networkClass):
    np.set_printoptions(threshold=np.inf)
    NeuNetwork.forwardPropagation(predictCl.data)
    m = len(NeuNetwork.A)
    result = NeuNetwork.A[m - 1]
    predictions = []
    probability = []
    for i in range(result.shape[1]):  # pour chaque sample
        predicNbr = 0
        proba = result[predicNbr][i]
        for nbr in range(result.shape[0]):  # pour chaque classe
            if result[nbr][i] > result[predicNbr][i]:
                predicNbr = nbr
                proba = result[predicNbr][i]
        predictions.append(predicNbr)
        probability.append(proba)
    precision = 0
    for i in range (len(predictions)):
        print (f"{predictions[i]}, {probability[i] *100:.2f}%, , {predictCl.result[i]}")
        if predictions[i] == predictCl.result[i]:
            precision += 1
    precision = precision / (len(predictions))
    print (f"Precision is : {precision * 100:.2f}%")

    
