import numpy as np
import math

class Batch():
    """A batch which contains a subset of the original dataset.
    """
    def __init__(self, data):
        self.data = data

    @staticmethod
    def divide_in_batches(X, Y, n) -> []:
        assert len(X) == len(Y)
        indices = np.random.permutation(len(X))
        X, Y = X[indices], Y[indices]

        def divide_arr_in_chunks(arr, chunk_size):
            return np.array([arr[i: i + n] for i in range(0, len(arr), n)])

        x = divide_arr_in_chunks(X, n)
        y = divide_arr_in_chunks(Y, n)
        return (x, y)
