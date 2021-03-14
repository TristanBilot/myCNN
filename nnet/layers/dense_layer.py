import numpy as np
from math import sqrt
from .layer import Layer

class DenseLayer(Layer):
    def __init__(self, units: int, activation: str = ''):
        super().__init__(activation)
        self.units = units
        self.is_shape_initialized = False

    def forward(self, X):
        if not self.is_shape_initialized:
            self.is_shape_initialized = True
            self._init_shape(X)

        forwarded = X.dot(self.weights) + self.bias
        self.shape = forwarded.shape
        return forwarded

    def _init_shape(self, X):
        # utiliser une classe pour X après
        in_channels = X.shape[X.ndim - 1]
        scale = 1 / sqrt(in_channels)
        self.weights = np.random.randn(in_channels, self.units) * scale
        self.bias = np.random.randn(1, self.units) * scale