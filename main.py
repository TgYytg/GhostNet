from neuron import Neuron
from trainer import Trainer
#Данные для обучения
data = [([0.150, 0, 0], 0), ([0.160, 0, 0], 0), ([0.170, 0, 0], 0), ([0.180, 1, 1], 1), ([0.190, 1, 1], 1)]

neuron = Neuron(3)

trainer = Trainer(0.1)

trainer.train(neuron, data, 100)


