import numpy as np
from layers import Layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class Activation(Layer):
    def __init__(self, activation: str = ''):
        super().__init__(activation)

    def forward(self, X):
        if self.activation == '':
            return X
        if self.activation == 'sigmoid':
            return sigmoid(X)
        if self.activation == 'relu':
            return relu(X)

class Sigmoid(Layer):
    def __init__(self, activation: str = ''):
        super().__init__(activation)

    def forward(self, X):
        return sigmoid(X)

    def backward(self, dy):
        pass
