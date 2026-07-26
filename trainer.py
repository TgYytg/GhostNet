from neuron import Neuron

class Trainer:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def mse(self, data, neuron):

        total = 0

        for x, target in data:
            prediction = neuron.forward(x)
            total += (target - prediction) ** 2

        return total / len(data)

    def train(self, neuron, data, epochs):

        for epoch in range(epochs):

            current_loss = self.mse(data, neuron)

            old_weight = neuron.weight

            neuron.weight += self.learning_rate

            new_loss = self.mse(data, neuron)

            if new_loss > current_loss:
                neuron.weight = old_weight - self.learning_rate

                new_loss = self.mse(data, neuron)

                if new_loss > current_loss:
                    neuron.weight = old_weight

            print(
                f"Epoch {epoch + 1}",
                f"weight={neuron.weight:.5f}",
                f"loss={self.mse(data, neuron):.5f}"
            )