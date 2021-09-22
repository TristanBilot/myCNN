import numpy as np

class TanH():
 
    def __init__(self, alpha = 1.7159):
        self._alpha = alpha
        self._cache = None

    def forward(
        self,
        X,
    ):
        self._cache = X
        return self._alpha * np.tanh(X)

    def backward(
        self,
        dy,
    ):
        X = self._cache
        return dy * (1 - np.tanh(X)**2)

class Softmax():
    
    def __init__(self):
        pass

    def forward(
        self,
        X,
    ):
        e_x = np.exp(X - np.max(X))
        return  e_x / np.sum(e_x, axis=1)[:, np.newaxis]

    def backward(
        self,
        dy,
        y,
    ):
        return dy - y
