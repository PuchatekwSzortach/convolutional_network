"""
A simple MNIST network with two hidden layers
"""

import tqdm
import numpy as np
import sklearn.utils
import sklearn.datasets
import sklearn.preprocessing

import net.layers
import net.models


def main():

    print("Loading data...")
    mnist = sklearn.datasets.fetch_mldata('MNIST original')

    X_train, y_train = mnist.data[:60000], mnist.target[:60000].reshape(-1, 1)
    X_test, y_test = mnist.data[60000:], mnist.target[60000:].reshape(-1, 1)

    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    encoder = sklearn.preprocessing.OneHotEncoder(sparse=False).fit(y_train)

    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    # Stack images so they are 3D, scale them to <0, 1> range
    X_train = np.stack([X_train], axis=-1) / 255
    X_test = np.stack([X_test], axis=-1) / 255

    layers = [
        net.layers.Input(sample_shape=(28, 28, 1)),
        net.layers.Convolution2D(nb_filter=20, nb_row=14, nb_col=14),
        net.layers.Convolution2D(nb_filter=10, nb_row=15, nb_col=15),
        net.layers.Flatten(),
        net.layers.Softmax()
    ]

    model = net.models.Model(layers)

    batch_size = 128
    epochs = 20

    X_test, y_test = sklearn.utils.shuffle(X_test, y_test)
    print("Initial accuracy: {}".format(model.get_accuracy(X_test, y_test)))

    for epoch in range(epochs):

        print("Epoch {}".format(epoch))

        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

        batches_count = len(X_train) // batch_size

        for batch_index in tqdm.tqdm(range(batches_count)):

            x_batch = X_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_batch = y_train[batch_index * batch_size: (batch_index + 1) * batch_size]

            model.train(x_batch, y_batch, learning_rate=0.01)

        print("Accuracy: {}".format(model.get_accuracy(X_test, y_test)))


if __name__ == "__main__":

    main()
