"""
Module with layers
"""

import numpy as np


class Layer:

    def __init__(self):

        self.input_shape = None
        self.output_shape = None

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

    def __init__(self, sample_shape):
        """
        Constructor
        :param sample_shape: 3 elements tuple (height, width, channels)
        """

        super().__init__()

        self.input_shape = (None,) + tuple(sample_shape)

    def build(self, input_shape=None):

        # Input layer's output shape is same as its input shape
        self.output_shape = self.input_shape

    def forward(self, x):

        if x.shape[1:] != self.input_shape[1:]:

            raise ValueError("{} shape expected, but input has shape {}".format(
                self.input_shape, x.shape))

        return x


# class Flatten(Layer):
#     """
#     Layer for flattening input. Outer dimension (batch size) is not affected
#     """
#
#     def __init__(self):
#
#         super().__init__()
#
#     def build(self, input_shape):
#
#         output_shape = []
#
#         for dimension in input_shape:
#
#             if dimension != 1:
#
#                 output_shape.append(dimension)
#
#         self.output_shape = output_shape
#
#     def forward(self, x):
#
#         squeezed = np.squeeze(x)
#
#         # If batch size was one, squeeze removed it, so add it back again
#         if x.shape[0] == 1:
#
#             squeezed = np.array([squeezed])
#
#         return squeezed
