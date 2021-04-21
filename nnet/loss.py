import numpy as np
from layer import Layer

class Loss(Layer):
    def forward(self, X, Y):
        pass

    def backward(self, dy):
        pass

class MeanSquare(Layer):
    def forward(self, X, Y):
        self.X = X
        return np.mean(np.sum((X - Y) ** 2, axis=1, keepdims=True))

    def backward(self, Y):
        N = self.X.shape[0]
        return 2 * (self.X - Y) / N
