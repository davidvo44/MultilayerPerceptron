
class Parameter:
    def __init__(self):
        self.learningRate = 0.1
        self.batchSize = 20
        self.epoch  = 500
        self.loss = "Categorical Crossentropy"
        self.earlyStop = True
        self.optimiser = "Standard"


    def __setattr__(self, name, value):
        self.__dict__[name] = value


    def __getattr__(self, item):
        return (self.__dict__[item])