"""
Module with models definitions
"""

import numpy as np


class Model:
    """
    Machine learning model, built from layers
    """

    def __init__(self, layers):
        """
        Constructor
        :param layers: list of Layers instances
        """

        self.layers = layers

        output_shape = None

        for layer in self.layers:

            layer.build(output_shape)
            output_shape = layer.output_shape

    def predict(self, x):

        for layer in self.layers:

            x = layer.forward(x)

        return x

    def get_loss(self, labels, predictions):
        """
        Compute categorical crossentropy loss
        :param labels: batch of labels
        :param predictions: batch of predictions
        :return: mean loss of the batch predictions
        """

        epsilon = 1e-7
        clipped_predictions = np.clip(predictions, epsilon, 1 - epsilon)

        return np.mean(-np.sum(labels * np.log(clipped_predictions), axis=1))

    def train(self, x, y, learning_rate):

        activation = x

        for layer in self.layers:

            activation = layer.train_forward(activation)

        gradients = self.layers[-1].get_output_layer_error_gradients(y)

        for layer in reversed(self.layers[:-1]):

            gradients = layer.train_backward(gradients, learning_rate)

    def get_accuracy(self, x, y):

        predictions = self.predict(x)

        correct_predictions_count = sum(
            [np.argmax(prediction) == np.argmax(ground_truth) for prediction, ground_truth in zip(predictions, y)])

        return correct_predictions_count / len(predictions)
