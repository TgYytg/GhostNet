from neuron import Neuron

class Trainer:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def train(self, neuron, data, epochs):

        for epoch in range(epochs):

            total_loss = 0

            for x, target in data:

                prediction = neuron.forward(x)

                loss = (target - prediction) ** 2
                total_loss += loss
                #Производная потерь по смещению
                d_loss_db = 2 * (prediction - target)
                #Производная потерь по весу
                d_loss_dw = d_loss_db * x
                #Градиентный спуск для обновления веса и смещения
                neuron.weight -= self.learning_rate * d_loss_dw
                neuron.bias -= self.learning_rate * d_loss_db
            average_loss = total_loss / len(data)

            print(f"Epoch {epoch + 1:3d} | "f"Loss = {average_loss:.6f} | "f"W = {neuron.weight:.6f} | "f"B = {neuron.bias:.6f}")
