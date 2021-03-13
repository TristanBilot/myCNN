import tensorflow as tf
from tensorflow import keras

import numpy as np
from layers.conv2D_layer import Conv2DLayer
from neural_net import NeuralNet
import matplotlib.pyplot as plt
import unittest

class NeuralNetTests(unittest.TestCase):

    def test_nn_batches1(self):
        nb_layers = 100
        nb_batches = 3
        X = np.random.randn(nb_layers, 32, 32, 3)
        nn = NeuralNet(layers=X, nb_batches=nb_batches)

        self.assertEqual(nn.batches[1].layers.shape, (nb_layers // nb_batches, *X[0].shape))
        self.assertEqual(nn.batches[1].nb_layers, nb_layers // nb_batches)

    def test_nn_batches2(self):
        nb_layers = 1000
        nb_batches = 32
        X = np.random.randn(nb_layers, 32, 32, 3)
        nn = NeuralNet(layers=X, nb_batches=nb_batches)
        sum = 0
        for b in nn.batches:
            sum += len(b.layers)

        self.assertEqual(sum, nb_layers)


if __name__ == "__main__":
    unittest.main()
