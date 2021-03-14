import numpy as np
from math import sqrt
from itertools import product
from .layer import Layer

class Conv2DLayer(Layer):
    def __init__(self, nb_filters, kernel=None, strides=(1, 1)):
        self.strides = strides
        self.nb_filters = nb_filters
                       
        if kernel != None and isinstance(kernel, (tuple, int)):
            self.kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        else:
            raise ValueError('The argument `kernel` should be set to a tuple or int value.')

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Applies `nb_filters` randomly generated trainable kernel convolutions
        of size `kernel` with `strides` offset.
        `X` should be in shape (N items from a batch, in-channels, height, width).
        """
        N, c, h, w = X.shape
        k1, k2 = self.kernel
        s1, s2 = self.strides
        output_shape = (N, self.nb_filters, (h - 2 * (k1 // 2)) // s1, (w - 2 * (k2 // 2)) // s2)
        output = np.ndarray(output_shape)
        self._set_weights(X)

        for n in range(N):
            for f in range(self.nb_filters): 
                for y, x in product(range(output_shape[2]), range(output_shape[3])):
                    area = X[n, :, y * s1 : y * s1 + k1, x * s2 : x * s2 + k2]
                    output[n, f, y, x] = np.sum(np.dot(self.weights[f], area))
            self._set_weights(X)
                
        self.shape = output_shape
        return output

    def _set_weights(self, X):
        in_channels = X.shape[X.ndim - 1]
        # update the weights of shape (channels_input, kern_x, kern_y, channels_output)
        self.weights = np.random.normal(size=(self.nb_filters, in_channels, *self.kernel))
        self.biases = np.zeros(shape=(self.nb_filters, 1))

