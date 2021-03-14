import tensorflow as tf
from tensorflow import keras

import numpy as np
from layers.conv2D_layer import Conv2DLayer
from layers.flatten import Flatten
from layers.dense_layer import DenseLayer
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
    print(X.shape)
    Y = Conv2DLayer(10, (3, 3)).forward(X)
    print(Y.shape)
    Y = Flatten().forward(Y)
    print(Y.shape)
    Y = DenseLayer(64).forward(Y)
    print(Y.shape)

    # Y = np.einsum('adbc->abcd', Y)
    # for i in range(10):
    #     disp_image(Y[i])