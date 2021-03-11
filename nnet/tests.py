import tensorflow as tf
from tensorflow import keras

import numpy as np
from layers.conv2D_layer import Conv2DLayer
import matplotlib.pyplot as plt

def disp_image(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    cifar10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    X = train_images[0:1]
    X = X / 255.0
    # disp_image(X[0])
    # X = np.einsum('klij->kjli', X)
    # X = np.random.randint(0, 10, (10, 32, 32, 3))
    Y = Conv2DLayer(6, (3, 3)).forward(X)
    Z = Conv2DLayer(64, (3, 3)).forward(Y)
    Z2 = Conv2DLayer(128, (3, 3)).forward(Z)
    # print(Y.shape)
    # Y = np.einsum('akli->alik', Y)
    for i in range(4):
        disp_image(Z2[i])
    # print(Y)
    print(Z2.shape)