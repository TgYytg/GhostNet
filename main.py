from neuron import Neuron
# данные для обучения нейрона
data = [(150, 0),(160, 0),(170, 0),(180, 1),(190, 1)]

learning_rate = 0.00001
neuron = Neuron(0, 0)
#процесс обучения
for epoch in range(100):
    total_error = 0

    for height, target in data:

        prediction = neuron.forward(height)
        error = target - prediction
        total_error += abs(error)
        neuron.weight += error * height * learning_rate

    average_error = total_error / len(data)

    print(epoch, average_error)

print(neuron.weight)  