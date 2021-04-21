import numpy as np
from layer import Layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1*(x > 0)

class Sigmoid(Layer):
    def forward(self, X):
        Y = sigmoid(X)
        self.shape = Y.shape
        return Y

    def backward(self, dy):
        self.gradient = sigmoid_derivative(dy)
        return dy * self.gradient

class ReLu(Layer):
    def forward(self, X):
        Y = relu(X)
        self.shape = Y.shape
        return Y

    def backward(self, dy):
        self.gradient = relu_derivative(dy)
        return dy * self.gradient
