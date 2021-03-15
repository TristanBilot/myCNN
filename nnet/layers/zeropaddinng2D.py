from .layer import Layer
import numpy as np
from itertools import product

class ZeroPadding2D(Layer):
    def __init__(self, padding=(1, 1), activation: str = ''):
        super().__init__(activation)
        self.padding = padding

    def forward(self, X: np.ndarray):
        dim = 4
        if X.ndim != dim:
            raise ValueError('ndim={dim} expected for `X` in ZeroPadding2D layer, found ndim={X.ndim}')

        output = np.pad(X, ((0,0), (0,0), self.padding, self.padding), 'constant', constant_values = (0,0))
        self.shape = output.shape
        return output