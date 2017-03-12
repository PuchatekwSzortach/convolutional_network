"""
A simple MNIST network with two hidden layers
"""
import time

import os

import tqdm
import numpy as np
import keras.datasets.mnist
import keras.utils.np_utils
import sklearn.utils

import net.layers
import net.models


def main():

    path = os.path.abspath("../../data/mnist/mnist.pkl.gz")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path)

    # Stack images so they are 3D, scale them to <0, 1> range
    X_train = np.stack([X_train], axis=-1) / 255
    X_test = np.stack([X_test], axis=-1) / 255

    y_train = keras.utils.np_utils.to_categorical(y_train)
    y_test = keras.utils.np_utils.to_categorical(y_test)

    layers = [
        net.layers.Input(sample_shape=(28, 28, 1)),
        net.layers.Convolution2D(nb_filter=20, nb_row=14, nb_col=14),
        net.layers.Convolution2D(nb_filter=10, nb_row=15, nb_col=15),
        net.layers.Flatten(),
        net.layers.Softmax()
    ]

    model = net.models.Model(layers)

    batch_size = 32
    epochs = 20

    print("Accuracy: {}".format(model.get_accuracy(X_test[:100], y_test[:100])))

    for epoch in range(epochs):

        print("Epoch {}".format(epoch))

        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

        batches_count = len(X_train[:1000]) // batch_size

        for batch_index in tqdm.tqdm(range(batches_count)):

            x_batch = X_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_batch = y_train[batch_index * batch_size: (batch_index + 1) * batch_size]

            model.train(x_batch, y_batch, learning_rate=0.01)

        print("Accuracy: {}".format(model.get_accuracy(X_test, y_test)))


if __name__ == "__main__":

    main()
