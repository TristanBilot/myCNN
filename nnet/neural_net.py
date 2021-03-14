from typing import Tuple, List
from layers.layer import Layer
from batch.batch import Batch
from layers.activation import Activation
from math import floor
import numpy as np

class NeuralNet():
    def __init__(self, loss: str=None, layers: List[Layer]=[]):
        self.loss = loss
        self.layers = layers

        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError('The argument `layers` should containns only `Layer` objects, not ', type(layer))

    def train(self, X, Y, epochs=1, batch_size=5):
        # Split X in `batch_size` batch subarrays.
        splitted_X = np.array_split(X, batch_size)
        self.batches = list(map(lambda x: Batch(len(x), x), splitted_X))

        for _ in range(epochs):
            for batch in self.batches:
                X = batch.X
                for layer in self.layers:
                    X = layer.forward(X)
                    X = Activation(layer.activation).forward(X)
                # make backpropagation with X

    def evaluate(self, X, Y):
        pass

    def predict(self, x):
        pass

    def add(self, layer: Layer):
        self.layers.append(layer)

    def summary(self):
        for layer in self.layers:
            print(layer.shape)
    
    # def _link_layers(self):
    #     for i in range(len(self.layers) - 1):
    #         next = self.layers[i + 1]
    #         next_layer = next.layer.shape[1 if i % 2 == 0 else 0]
    #         self.layers[i].set_output_layer(next_layer)
