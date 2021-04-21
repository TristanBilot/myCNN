from typing import Tuple, List
from layer import Layer
from batch import Batch
from math import floor
import numpy as np

class NeuralNet():
    def __init__(self, layers: List[Layer]=[]):
        self.layers = layers

        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError('The argument `layers` should containns only `Layer` objects, not ', type(layer))

    def train(self, X, Y, loss, epochs=1, batch_size=5, learning_rate=0.1):
        (self.X_batches, self.Y_batches) = Batch.divide_in_batches(X, Y, batch_size)

        for _ in range(epochs):
            for (X, Y) in zip(self.X_batches, self.Y_batches):

                # forward pass
                for layer in self.layers:
                    X = layer.forward(X)

                self.summary()

                # backward pass
                dy = loss.forward(X, Y)
                gradient = loss.backward(Y)
                print(f'dy =====> ${dy}')
                print(type(gradient))
                print(gradient)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)

                # update weights with the gradients
                for layer in self.layers:
                    layer.update_weights(learning_rate)


    def evaluate(self, X, Y):
        pass

    def predict(self, x):
        pass

    def add(self, layer: Layer):
        self.layers.append(layer)

    def summary(self):
        for layer in self.layers:
            print(layer.shape)
    