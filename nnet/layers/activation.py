import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.layer import Layer
from activation.functions import sigmoid, relu

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
