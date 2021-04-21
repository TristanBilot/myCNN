import numpy as np
from math import sqrt
from itertools import product

class Layer():
    def __init__(self):
        self.X = None
        self.shape = None
        self.gradient = None
        self.weights = None
        self.bias = None

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def update_weights(self, learning_rate):
        self.weights = self.weights - (learning_rate * self.gradient)

class Dense(Layer):
    def __init__(self, units: int, activation: str = None):
        self.units = units
        self.is_shape_initialized = False

    def forward(self, X):
        if not self.is_shape_initialized:
            self.is_shape_initialized = True
            self._init_shape(X)


        
        Y = np.dot(X, self.weights) + self.bias
        print(X.shape)
        print(Y.shape)
        print('---eyeyey')
        self.shape = Y.shape
        self.X = X
        return Y

    def backward(self, dy):
        print(self.X.shape)
        print(dy.shape)
        dw = np.dot(self.X.T, dy)
        dx = np.dot(dy, self.weights.T)
        self.gradient = dw
        return dx

        # self.bias.gradient += np.sum(dy, axis=0, keepdims=True)

    def _init_shape(self, X):
        # utiliser une classe pour X après
        in_channels = X.shape[X.ndim - 1]
        scale = 1 / sqrt(in_channels)
        self.weights = np.random.randn(in_channels, self.units) * scale
        self.bias = np.random.randn(1, self.units) * scale

class Flatten(Layer):
    def forward(self, X):
        self.old_shape = X.shape
        batch_size = X.shape[0]
        forwarded = X.reshape(batch_size, -1)
        self.shape = forwarded.shape
        return forwarded

    def backward(self, dy):
        print(f'shape: {self.old_shape} + {dy.shape} ')
        return dy.reshape(self.old_shape)

class Conv2D(Layer):
    def __init__(self, nb_filters, kernel=None, strides=(1, 1), activation: str = None):
        self.strides = strides
        self.nb_filters = nb_filters

        if kernel != None and isinstance(kernel, (tuple, int)):
            self.kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        else:
            raise ValueError('The argument `kernel` should be set to a tuple or int value.')

    def forward(self, X) -> np.ndarray:
        """Applies `nb_filters` randomly generated trainable kernel convolutions
        of size `kernel` with `strides` offset.
        `X` should be in shape (N items from a batch, in-channels, height, width).
        """
        N, h, w, c = X.shape
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

class PoolingLayer(Layer):
    def __init__(self, pooling_method: str, kernel, strides, activation: str):
        self.pooling_method = pooling_method

        if not isinstance(kernel, (tuple, int)):
            raise ValueError('The argument `kernel` should be set to a tuple or int value.')
        self.kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)

        if not isinstance(strides, (tuple, int)):
            raise ValueError('The argument `strides` should be set to a tuple or int value.')
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)

    def forward(self, X):
        N, h, w, c = X.shape
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

class ZeroPadding2D(Layer):
    def __init__(self, padding=(1, 1), activation: str = None):
        self.padding = padding

    def forward(self, X: np.ndarray):
        dim = 4
        if X.ndim != dim:
            raise ValueError('ndim={dim} expected for `X` in ZeroPadding2D layer, found ndim={X.ndim}')

        output = np.pad(X, ((0,0), (0,0), self.padding, self.padding), 'constant', constant_values = (0,0))
        self.shape = output.shape
        return output

class MinPool2D(PoolingLayer):
    def __init__(self, kernel, strides=(1, 1), activation: str = None):
        super().__init__('min', kernel, strides, activation)

    def forward(self, X):
        return super().forward(X)

class MaxPool2D(PoolingLayer):
    def __init__(self, kernel, strides=(1, 1), activation: str = None):
        super().__init__('max', kernel, strides, activation)

    def forward(self, X):
        return super().forward(X)

class MeanPool2D(PoolingLayer):
    def __init__(self, kernel, strides=(1, 1), activation: str = None):
        super().__init__('mean', kernel, strides, activation)

    def forward(self, X):
        return super().forward(X)