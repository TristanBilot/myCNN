import tensorflow as tf
from tensorflow import keras

import numpy as np
from layers.conv2D import Conv2D
from layers.flatten import Flatten
from layers.dense import Dense
from neural_net import NeuralNet
import matplotlib.pyplot as plt

def disp_image(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    cifar10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    X = train_images[0:10]
    X = X / 255.0

    # for i in range(10):
    #     disp_image(X[i])

    X = np.einsum('abcd->adbc', X)

    nn = NeuralNet('')
    nn.add(Conv2D(10, (3, 3), activation='relu'))
    nn.add(Conv2D(32, (3, 3)))
    nn.add(Flatten(activation='sigmoid'))
    nn.add(Dense(64))

    nn.train(X, None, batch_size=2)
    nn.summary()

    # Y = np.einsum('adbc->abcd', Y)
    # for i in range(10):
    #     disp_image(Y[i])