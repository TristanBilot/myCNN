from .layer import Layer
import numpy as np
from itertools import product

class PoolingLayer(Layer):
    def __init__(self, pooling_method: str, kernel, strides, activation: str):
        super().__init__(activation)
        self.pooling_method = pooling_method

        if not isinstance(kernel, (tuple, int)):
            raise ValueError('The argument `kernel` should be set to a tuple or int value.')
        self.kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)

        if not isinstance(strides, (tuple, int)):
            raise ValueError('The argument `strides` should be set to a tuple or int value.')
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)

    def forward(self, X):
        N, c, h, w = X.shape
        k1, k2 = self.kernel
        s1, s2 = self.strides
        output_shape = (N, c, h // (k1 * s1), w // (k2 * s2))
        output = np.ndarray(output_shape)

        for y, x in product(range(h, w)):
            area = X[:, :, y * s1: y * s1 + k1, x * s2: x * s2 + k2]
            output[:, :, x, y] = self._map_pooling_method(area, axis=(2, 3))

        self.shape = output_shape
        return output

    def _map_pooling_method(self, area, axis):
        if self.pooling_method == 'min':
            return np.min(area, axis)
        if self.pooling_method == 'max':
            return np.max(area, axis)
        if self.pooling_method == 'mean':
            return np.mean(area, axis)
        raise ValueError('The argument `pooling_method` is unknown. Valid methods are `min`, `max`, `mean`.')