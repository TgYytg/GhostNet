from neuron import Neuron
# данные для обучения нейрона
data = [(0.150, 0),(0.160, 0),(0.170, 0),(0.180, 1),(0.190, 1)]

learning_rate = 0.00001
neuron = Neuron(0, 0)
total_loss = 0
def mse(target, prediction):
    return (target - prediction) **2
#процесс обучения
for x, target in data:
    total_error = 0

    for height, target in data:

        prediction = neuron.forward(x)
        loss = mse(target, prediction)
        error = target - prediction
        
        total_error += abs(error)
        total_loss += loss
        neuron.weight += error * height * learning_rate

    average_error = total_error / len(data)

     print(
        f"x={x} "
        f"prediction={prediction:.3f} "
        f"target={target} "
        f"loss={loss:.3f}"
    )

print("Average loss:", total_loss / len(data))
