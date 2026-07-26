import math 

class Trainer:

    def train():
        pass

class Neuron:

    def __init__(self, weight = 0.00, bias = 0.00):
        self.weight = weight
        self.bias = bias


    def forward(self,x):
        return x * self.weight + self.bias
