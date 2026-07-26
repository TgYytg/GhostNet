import math 

class Trainer:

    def train():
        pass

class Neuron:

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias


    def forward(self,x):
        return x * self.weight + self.bias

    
neuron = Neuron(0.01, -1.8)

