"""
Module with layers
"""

import numpy as np


class Layer:

    def __init__(self):

        self.input_shape = None
        self.output_shape = None

        self.last_input = None
        self.last_output = None

    def build(self, input_shape):

        raise NotImplementedError()

    def forward(self, x):

        raise NotImplementedError()

    def train_forward(self, x):

        raise NotImplementedError()

    def train_backward(self, gradients, learning_rate=None):

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

    def train_forward(self, x):

        return self.forward(x)

    def train_backward(self, gradients, learning_rate=None):
        """
        For Input layer train backward is a no-op
        :return: None
        """

        return None


class Flatten(Layer):
    """
    Layer for flattening input. Outer dimension (batch size) is not affected
    """

    def __init__(self):

        super().__init__()

    def build(self, input_shape):

        self.input_shape = tuple(input_shape)

        squeezed_sample_shape = []

        for dimension in input_shape[1:]:

            if dimension != 1:

                squeezed_sample_shape.append(dimension)

        self.output_shape = (input_shape[0],) + tuple(squeezed_sample_shape)

    def forward(self, x):

        shape = (x.shape[0],) + self.output_shape[1:]
        return x.reshape(shape)

    def train_forward(self, x):

        self.last_input = x
        self.last_output = self.forward(x)

        return self.last_output

    def train_backward(self, gradients, learning_rate=None):

        return gradients.reshape(self.last_input.shape)


class Convolution2D(Layer):

    def __init__(self, nb_filter, nb_row, nb_col):
        """
        2D convolutional layer. Applies ReLU activation.
        :param nb_filter: number of filters
        :param nb_row: width of each filter
        :param nb_col: height of each filter
        """

        super().__init__()

        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col

        self.kernels = None
        self.biases = None

    def build(self, input_shape):

        if len(input_shape) != 4:

            raise ValueError("Input shape {} isn't 4-dimensional".format(input_shape))

        self.input_shape = input_shape
        self.output_shape = (None, input_shape[1] - self.nb_row + 1, input_shape[2] - self.nb_col + 1, self.nb_filter)

        input_channels = input_shape[-1]
        kernels_shape = (self.nb_filter, self.nb_row, self.nb_col, input_channels)

        scale = np.sqrt(2 / (self.nb_row * self.nb_col * input_channels))
        self.kernels = np.random.uniform(low=-scale, high=scale, size=kernels_shape)

        self.biases = np.zeros((self.nb_filter,), dtype=np.float32)

    def forward(self, x):

        preactivation = self.get_preactivation(x)

        # Apply ReLU activation
        return self.relu(preactivation)

    def get_preactivation(self, x):

        preactivation = np.zeros(shape=(x.shape[0],) + self.output_shape[1:])

        for sample_index, sample in enumerate(x):

            for kernel_index, (kernel, bias) in enumerate(zip(self.kernels, self.biases)):

                for row_index in range(0, x.shape[1] - self.nb_row + 1):

                    for col_index in range(0, x.shape[2] - self.nb_col + 1):
                        input_patch = sample[row_index: row_index + self.nb_row, col_index: col_index + self.nb_col, :]

                        preactivation[sample_index, row_index, col_index, kernel_index] = np.sum(input_patch * kernel) + bias

        return preactivation

    def relu(self, x):

        return x * (x > 0)

    def relu_derivative(self, x):

        return 1 * (x > 0)

    def train_forward(self, x):

        self.last_input = x

        preactivation = self.get_preactivation(x)
        self.last_output = self.relu(preactivation)

        return self.last_output

    def train_backward(self, gradients, learning_rate):

        preactivation_error_gradients = gradients * self.relu_derivative(gradients)

        for kernel_index in range(len(self.kernels)):

            self.update_bias(preactivation_error_gradients, kernel_index, learning_rate)
            self.update_kernel_weights(preactivation_error_gradients, kernel_index, learning_rate)

    def update_bias(self, preactivation_error_gradients, kernel_index, learning_rate):

        bias_error_gradients = preactivation_error_gradients[:, kernel_index, :, :]

        bias_error_gradients_sums = np.sum(bias_error_gradients, axis=(1, 2))
        mean_bias_error_gradient = np.mean(bias_error_gradients_sums)

        self.biases[kernel_index] -= learning_rate * mean_bias_error_gradient

    def update_kernel_weights(self, preactivation_error_gradients, kernel_index, learning_rate):

        # Iterating over all weights of the kernel
        for y in range(self.nb_row):

            input_row_start = y
            input_row_end = self.input_shape[1] - self.nb_row + y + 1

            for x in range(self.nb_col):

                input_column_start = x
                input_column_end = self.input_shape[2] - self.nb_col + x + 1

                for z in range(self.input_shape[-1]):

                    # Get part of preactivation_error_gradients that were affected by that weight
                    input_patch = self.last_input[
                                  :,
                                  input_row_start: input_row_end,
                                  input_column_start: input_column_end,
                                  kernel_index]

                    weight_errors_gradients = preactivation_error_gradients * input_patch

                    weight_errors_gradients_sum = np.sum(weight_errors_gradients, axis=(1, 2))
                    mean_weight_errors_gradient = np.mean(weight_errors_gradients_sum)

                    self.kernels[kernel_index, y, x, z] -= learning_rate * mean_weight_errors_gradient


class Softmax:
    """
    Softmax layer
    """

    def __init__(self):

        self.input_shape = None
        self.output_shape = None

        self.last_output = None

    def build(self, input_shape):

        if len(input_shape) != 2:

            message = "Input shape must be 2D, but {}D input with shape {} was given".format(
                len(input_shape), input_shape)

            raise ValueError(message)

        if input_shape[1] == 1:

            message = "Last input shape dimension must be larger than 1, but provided input shape is: {}"\
                .format(input_shape)

            raise ValueError(message)

        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, x):

        if x.shape[1:] != self.input_shape[1:]:

            message = "Input shape incorrect, expected {} but {} was given".format(
                self.input_shape, x.shape)

            raise ValueError(message)

        # Clip values to sensible range for numerical stability
        clipped = np.clip(x, -50, 50)

        exponentials = np.exp(clipped)
        exponential_sums = np.sum(exponentials, axis=1).reshape((x.shape[0], 1))

        return exponentials / exponential_sums

    def train_forward(self, x):

        self.last_output = self.forward(x)
        return self.last_output

    def get_output_layer_error_gradients(self, labels):
        """
        Given correct labels for last predictions, compute error gradients
        :param labels: batch of correct labels
        :return: batch of error gradients
        """

        return self.last_output - labels

