import numpy as np
from math import sqrt
from itertools import product
from .layer import Layer

class Conv2DLayer(Layer):
    def __init__(self, nb_filters, kernel=None, strides=(1, 1)):
        self.strides = strides
        self.nb_filters = nb_filters
        self.nb_channels = 3 # replace later
                       
        if kernel != None and isinstance(kernel, (tuple, int)):
            self.kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        else:
            raise ValueError('The argument `kernel` should be set to a tuple or int value.')
        self._set_weights()

    def forward(self, X) -> np.ndarray:
        N, h, w, c = X.shape
        k1, k2 = self.kernel
        s1, s2 = self.strides
        output_shape = (self.nb_filters, (h - 2 * (k1 // 2)) // s1, (w - 2 * (k2 // 2)) // s2, c)
        output = np.ndarray(output_shape)

        for n in range(self.nb_filters):
            for y, x in product(range(output_shape[1]), range(output_shape[2])):
                for z in range(c):
                    area = X[0, y * s1 : y * s1 + k1, x * s2 : x * s2 + k2, :]
                    output[n, y, x, z] = np.sum(self.weights[z] * area)
            self._set_weights()
        self.shape = output_shape
        return output

    def _set_weights(self):
        # update the weights of shape (channels_input, kern_x, kern_y, channels_output)
        self.weights = np.random.normal(size=(self.nb_channels, *self.kernel, self.nb_channels))
        self.biases = np.zeros(shape=(self.nb_channels, 1))

    # def forward(self, X) -> np.ndarray:
    #     N, C, H, W = X.shape
    #     KH, KW = self.kernel
    #     s1, s2 = self.strides[0], self.strides[1]
    #     out_shape = (N, self.nb_channels, 1 + (H - KH)//s1, 1 + (W - KW)//s2)
    #     Y = np.zeros(out_shape)
    #     for n in range(N):
    #         for c_w in range(self.nb_channels):
    #             for h, w in product(range(out_shape[2]), range(out_shape[3])):
    #                 h_offset, w_offset = h*s1, w*s2
    #                 rec_field = X[n, :, h_offset:h_offset + KH, w_offset:w_offset + KW]
    #                 Y[n, c_w, h, w] = np.sum(self.weight['W'][c_w]*rec_field) + self.weight['b'][c_w]

    #     return Y


