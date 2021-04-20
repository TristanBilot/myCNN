class Batch():
    """A batch which contains a sub-part of the original X dataset.
    """
    def __init__(self, nb_layers, X):
        self.X = X
