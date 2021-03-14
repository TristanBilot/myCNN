import numpy as np

class Layer():
    def __init__(self, activation: str):
        self.activation = activation

    def forward(self, X: np.ndarray):
        pass
    