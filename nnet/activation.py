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
        self.debug('FW: sigmoid', X.shape, Y.shape)
        return Y

    def backward(self, dy):
        self.gradient = sigmoid_derivative(dy)
        dx = dy * self.gradient
        self.debug('BW: sigmoid', dy.shape, dx.shape)
        return dx

    def update_weights(self, learning_rate):
        pass

class ReLu(Layer):
    def forward(self, X):
        Y = relu(X)
        self.shape = Y.shape
        return Y

    def backward(self, dy):
        self.gradient = relu_derivative(dy)
        return dy * self.gradient

    def update_weights(self, learning_rate):
        pass
