"""
Few MNIST networks based on Keras
"""

import os

import tqdm
import keras
import keras.datasets.mnist
import keras.utils.np_utils
import numpy as np
import sklearn.utils


def main():

    path = os.path.abspath("../../data/mnist/mnist.pkl.gz")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path)

    # Stack images so they are 3D, scale them to <0, 1> range
    X_train = np.stack([X_train], axis=-1) / 255
    X_test = np.stack([X_test], axis=-1) / 255

    y_train = keras.utils.np_utils.to_categorical(y_train)
    y_test = keras.utils.np_utils.to_categorical(y_test)

    input_shape = [28, 28, 1]

    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Convolution2D(nb_filter=20, nb_row=14, nb_col=14, activation='relu')(input)
    x = keras.layers.Convolution2D(nb_filter=10, nb_row=15, nb_col=15, activation='relu')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation('softmax')(x)

    model = keras.models.Model(input, x)

    optimizer = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Accuracy: {}".format(model.test_on_batch(X_test[:100], y_test[:100])[1]))

    batch_size = 32
    epochs = 20

    for epoch in range(epochs):

        print("Epoch {}".format(epoch))

        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

        # batches_count = len(X_train) // batch_size
        batches_count = 100

        for batch_index in tqdm.tqdm(range(batches_count)):

            x_batch = X_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_batch = y_train[batch_index * batch_size: (batch_index + 1) * batch_size]

            model.train_on_batch(x_batch, y_batch)

        print("Accuracy: {}".format(model.test_on_batch(X_test[:1000], y_test[:1000])[1]))


if __name__ == "__main__":

    main()