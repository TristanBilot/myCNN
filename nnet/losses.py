import numpy as np
from layers import Layer

class MeanSquare(Layer):
    def __init__(self, activation: str = ''):
        super().__init__(activation)

    def forward(self, X, Y):
        self.X = X
        return np.mean(np.sum((X - Y) ** 2, axis=1, keepdims=True))

    def backward(self, dy):
        N = self.X.shape[0]
        return 2 * (self.X - dy) / N
