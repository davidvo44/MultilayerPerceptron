
class Parameter:
    def __init__(self):
        self.learningRateFormat = "default"
        self.learningRate = 0.1
        self.batchSize = 20
        self.epoch  = 500
        self.loss = "categorical Crossentropy"


    def __setattr__(self, name, value):
        self.__dict__[name] = value


    def __getattr__(self, item):
        return (self.__dict__[item])