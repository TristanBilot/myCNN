import tensorflow as tf
from tensorflow import keras

import numpy as np
from layer import *
from activation import Sigmoid, ReLu
from neural_net import NeuralNet
from loss import MeanSquare
import matplotlib.pyplot as plt

def disp_image(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    cifar10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    X, Y = train_images[0:5], train_labels[0:5]
    X, Y = X / 255.0, Y / 255.0

    # for i in range(10):
    #     disp_image(X[i])

    # X = np.einsum('abcd->adbc', X)
    # print(X[0].shape)

    nn = NeuralNet()
    # nn.add(Conv2D(10, (3, 3)))
    # nn.add(ReLu())
    # nn.add(Conv2D(12, (3, 3)))
    # nn.add(Sigmoid())
    # nn.add(MaxPool2D((5, 5)))
    # nn.add(Sigmoid())
    # nn.add(ZeroPadding2D())
    # nn.add(Flatten())
    # nn.add(Dense(64))
    # nn.add(Sigmoid())

    nn.add(Dense(32))
    nn.add(Sigmoid())
    nn.add(Dense(64))
    nn.add(Sigmoid())
    nn.add(Flatten())
    nn.add(Dense(10))

    nn.train(X, Y, batch_size=3, loss=MeanSquare(), epochs=10)
    nn.summary()

    # Y = np.einsum('adbc->abcd', Y)
    # for i in range(10):
    #     disp_image(Y[i])