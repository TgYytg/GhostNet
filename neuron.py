import math 

class Neuron:

    def __init__(self, input_size):
        self.weights = [0.0] * input_size
        self.bias = 0.0


    def forward(self,inputs):
        output = self.bias

        for x,w in zip(inputs, self.weights):
            output += x * w
        return output


