from typing import Tuple, List
from layers.layer import Layer

class NeuralNet():
    def __init__(self, loss: str=None, layers: List[Layer]=None):
        self.loss = loss
        self.layers = layers if layers != None else []

    def train(self, x, y, epochs=1, batch_size=None):
        pass

    def evaluate(self, x, y):
        pass

    def predict(self, x):
        pass

    def add(self, layer: Layer):
        self.layers.add(layer)
    
    def _link_layers(self):
        for i in range(len(self.layers) - 1):
            next = self.layers[i + 1]
            next_layer = next.layer.shape[1 if i % 2 == 0 else 0]
            self.layers[i].set_output_layer(next_layer)
