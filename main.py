from neuron import Neuron
from trainer import Trainer
#Данные для обучения
data = [
    (0.150, 0),
    (0.160, 0),
    (0.170, 0),
    (0.180, 1),
    (0.190, 1),
]

neuron = Neuron()

trainer = Trainer(0.001)

trainer.train(neuron, data, 100)