from typing import Tuple, List
from layers.layer import Layer
from batch.batch import Batch
from math import floor
import numpy as np

class NeuralNet():
    def __init__(self, loss: str=None, layers: List[Layer]=[]):
        self.loss = loss
        self.layers = layers

    def train(self, X, Y, epochs=1, batch_size=5):
        # Split X in `batch_size` batch subarrays.
        splitted_X = np.array_split(X, batch_size)
        self.batches = list(map(lambda x: Batch(len(x), x), splitted_X))

        for i in range(epochs):
            for batch in self.batches:
                for layer in self.layers:
                    X = layer.forward(batch)
                # make backpropagation with X
        pass

    def evaluate(self, X, Y):
        pass

    def predict(self, x):
        pass

    def add(self, layer: Layer):
        self.layers.add(layer)
    
    # def _link_layers(self):
    #     for i in range(len(self.layers) - 1):
    #         next = self.layers[i + 1]
    #         next_layer = next.layer.shape[1 if i % 2 == 0 else 0]
    #         self.layers[i].set_output_layer(next_layer)
