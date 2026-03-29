import networkClass

def predict(predictCl, NeuNetwork: networkClass):
    NeuNetwork.forwardPropagation(predictCl.data)
    m = len(NeuNetwork.A)
    result = NeuNetwork.A[m - 1]
    print (result)
    print (f"Real: {predictCl.result}")