from .layer import Layer

class Flatten(Layer):
    def __init__(self, activation: str = ''):
        super().__init__(activation)
        
    def forward(self, X):
        batch_size = X.shape[0]
        forwarded = X.reshape(batch_size, -1)
        self.shape = forwarded.shape
        return forwarded
