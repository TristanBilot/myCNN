import numpy as np

class DenseLayer():
    def __init__(self, size: int):
        self.size = size

    def set_output_layer(self, output):
        self.layer = np.random.randn(self.size, output)