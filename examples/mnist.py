"""
A few MNIST networks
"""

import os
import numpy as np
import keras.datasets.mnist
import sklearn.preprocessing


def main():

    path = os.path.abspath("../../data/mnist/mnist.pkl.gz")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path)

    # Stack images so they are 3D, scale them to <0, 1> range
    X_train = np.stack([X_train], axis=-1) / 255
    X_test = np.stack([X_test], axis=-1) / 255

    labels_binarizer = sklearn.preprocessing.LabelBinarizer().fit(y_test)
    y_train = labels_binarizer.transform(y_train)
    y_test = labels_binarizer.transform(y_test)


if __name__ == "__main__":

    main()
