# Dot product

import numpy as np

inputs = [1, 2, 3, 2.5]

# better to use 2D array than multiple 1D arrays
weights = [[0.2, 0.8, -0.5, 1], 
        [0.5, -0.91, 0.26, -0.5], 
        [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

# using dot product XaXb + YaYb.. it's same as |a|*|b|cos(theta) from cosine theory
layer_outputs = np.dot(weights, inputs) + bias

print(layer_outputs)