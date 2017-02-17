import os

import keras.datasets.mnist
import numpy as np
import sklearn.preprocessing
import cv2


def get_single_layer_model(input_shape):
    """
    7850 parameters.
    Should get about 83%~92% accuracy
    """

    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Convolution2D(nb_filter=10, nb_row=28, nb_col=28, activation='relu')(input)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation('softmax')(x)

    return keras.models.Model(input, x)


def get_two_layers_model(input_shape):
    """
    8280 parameters
    Should get about 74%~96% accuracy
    """

    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Convolution2D(nb_filter=10, nb_row=4, nb_col=4, subsample=(3, 3), activation='relu')(input)
    x = keras.layers.Convolution2D(nb_filter=10, nb_row=9, nb_col=9, activation='relu')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation('softmax')(x)

    return keras.models.Model(input, x)


def get_three_layers_model(input_shape):
    """
    1990 parameters
    Should get about 64~95% accuracy
    """

    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Convolution2D(nb_filter=10, nb_row=4, nb_col=4, subsample=(3, 3), activation='relu')(input)
    x = keras.layers.Convolution2D(nb_filter=10, nb_row=3, nb_col=3, subsample=(3, 3), activation='relu')(x)
    x = keras.layers.Convolution2D(nb_filter=10, nb_row=3, nb_col=3, activation='relu')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation('softmax')(x)

    return keras.models.Model(input, x)


def main():

    path = os.path.abspath("../../data/mnist/mnist.pkl.gz")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path)

    # Stack images so they are 3D, scale them to <0, 1> range
    X_train = np.stack([X_train], axis=-1) / 255
    X_test = np.stack([X_test], axis=-1) / 255

    labels_binarizer = sklearn.preprocessing.LabelBinarizer().fit(y_test)
    y_train = labels_binarizer.transform(y_train)
    y_test = labels_binarizer.transform(y_test)

    input_shape = [28, 28, 1]

    # model = get_single_layer_model(input_shape)
    # model = get_two_layers_model(input_shape)
    model = get_three_layers_model(input_shape)

    # print(model.predict(X_test[:20]).shape)

    optimizer = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, nb_epoch=20, batch_size=32, validation_data=(X_test, y_test))


if __name__ == "__main__":

    main()