from .layer import Layer

class Flatten(Layer):
    def forward(self, X):
        batch_size = X.shape[0]
        forwarded = X.reshape(batch_size, -1)
        self.shape = forwarded.shape
        return forwarded
