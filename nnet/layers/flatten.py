from .layer import Layer

class Flatten(Layer):
    def forward(self, X):
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
