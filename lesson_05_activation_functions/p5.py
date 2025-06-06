# Implementing ReLU activation

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3) # spiral data with 2 features

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU: # Rectified Linear Unit Activation
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)