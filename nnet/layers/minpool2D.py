from .layer import Layer
from .pooling_layer import PoolingLayer

class MinPool2D(PoolingLayer):
    def __init__(self, kernel, strides=(1, 1), activation: str = ''):
        super().__init__('min', kernel, strides, activation)

    def forward(self, X):
        return super().forward(X)