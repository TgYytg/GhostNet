from neuron import Neuron

class Layer:
    def __init__(self, num_neurons):
        self.neurons = []

        for _ in range(num_neurons):
            self.neurons.append(Neuron())

    def forward(self,x):
        outputs = []

        for neuron in self.neurons:
            outputs.append(neuron.forward(x))
        return outputs
        