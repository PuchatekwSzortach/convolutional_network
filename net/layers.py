"""
Module with layers
"""

import numpy as np


class Layer:

    def __init__(self):

        self.output_shape = None
        self.activation = None

    def build(self, input_shape):

        raise NotImplementedError()

    def forward(self, x):

        raise NotImplementedError()

    def train_forward(self):

        raise NotImplementedError()

    def train_backward(self):

        raise NotImplementedError()


class Input(Layer):
    """
    Input layer
    """

    def __init__(self, shape):
        """
        Constructor
        :param shape: 3 elements tuple (height, width, channels)
        """

        super().__init__()

        if len(shape) != 3:

            raise ValueError("Shape must have 3 elements, but {} were provided".format(len(shape)))

        are_dimensions_integers = [isinstance(x, int) for x in shape]

        if not all(are_dimensions_integers):

            raise ValueError("Not all elements of shape {} are integers".format(shape))

        self.shape = shape

    def build(self, input_shape):

        # Input layer's output shape is defined in its constructor
        self.output_shape = self.shape

    def forward(self, x):

        return x


class Flatten(Layer):
    """
    Layer for flattening input. Outer dimension (batch size) is not affected
    """

    def __init__(self):

        super().__init__()

    def build(self, input_shape):

        output_shape = []

        for dimension in input_shape:

            if dimension != 1:

                output_shape.append(dimension)

        self.output_shape = output_shape

    def forward(self, x):

        squeezed = np.squeeze(x)

        # If batch size was one, squeeze removed it, so add it back again
        if x.shape[0] == 1:

            squeezed = np.array([squeezed])

        return squeezed
