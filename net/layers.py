"""
Module with layers
"""

import itertools

import numpy as np
import net.conversions


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

    def __init__(self, filters, rows, columns):
        """
        2D convolutional layer. Applies ReLU activation.
        :param filters: number of filters
        :param rows: number of rows of each filter
        :param columns: number of columns of each filter
        """

        super().__init__()

        self.filters = filters
        self.rows = rows
        self.columns = columns

        self.kernels = None
        self.biases = None

        self.last_preactivation = None

    def build(self, input_shape):

        if len(input_shape) != 4:

            raise ValueError("Input shape {} isn't 4-dimensional".format(input_shape))

        self.input_shape = input_shape

        # Dimensions are [images, rows, columns, filters]
        self.output_shape = (None, input_shape[1] - self.rows + 1, input_shape[2] - self.columns + 1, self.filters)

        input_channels = input_shape[-1]
        kernels_shape = (self.filters, self.rows, self.columns, input_channels)

        scale = np.sqrt(2 / (self.rows * self.columns * input_channels))
        self.kernels = np.random.uniform(low=-scale, high=scale, size=kernels_shape)

        self.biases = np.zeros((self.filters,), dtype=np.float32)

    def forward(self, input):

        preactivation = self._get_preactivation(input)

        # Apply ReLU activation
        return self._relu(preactivation)

    def _get_preactivation(self, input):

        # Reshape kernels into a matrix so that each column stands for a single flattened kernel
        kernels_matrix = self.kernels.reshape(self.kernels.shape[0], -1).T

        # Get image batches matrix representation such that each image patch that gets convolved is represented
        # by a single matrix row. After one image ends, second starts and so on
        image_batches_matrix = net.conversions.get_images_batch_patches_matrix(input, self.kernels.shape[1:])

        # Compute responses of all patches with kernel
        # Response is a 2D matrix with dimensions: patches, kernels
        # E.g each row represents results of convolving a single patch with all kernels
        response = np.dot(image_batches_matrix, kernels_matrix)

        # Reshape response to 4D images batch with dimensions image y, x, output channels
        response_shape = (input.shape[0], input.shape[1] - self.kernels.shape[1] + 1,
                          input.shape[2] - self.kernels.shape[2] + 1, self.kernels.shape[0])

        response = response.reshape(response_shape)

        preactivation = response + self.biases

        return preactivation

    def _relu(self, x):

        return x * (x > 0)

    def _relu_derivative(self, x):

        return 1 * (x > 0)

    def train_forward(self, input):

        self.last_input = input

        self.last_preactivation = self._get_preactivation(input)
        self.last_output = self._relu(self.last_preactivation)

        return self.last_output

    def train_backward(self, gradients, learning_rate):

        preactivation_error_gradients = gradients * self._relu_derivative(self.last_preactivation)

        # Copy old kernels, we will use original values when computing image gradients
        old_kernels = self.kernels.copy()

        self._update_kernels(preactivation_error_gradients, learning_rate)
        self._update_biases(preactivation_error_gradients, learning_rate)

        return self._get_image_gradients(preactivation_error_gradients, old_kernels)

    def _update_biases(self, preactivation_error_gradients, learning_rate):

        for kernel_index in range(len(self.kernels)):

            # Sum contributions for each image
            bias_error_gradients_sums = np.sum(preactivation_error_gradients[:, :, :, kernel_index], axis=(1, 2))

            # And take mean across contributions to different images
            mean_bias_error_gradient = np.mean(bias_error_gradients_sums)

            self.biases[kernel_index] -= learning_rate * mean_bias_error_gradient

    def _update_kernels(self, preactivation_error_gradients, learning_rate):

        kernels_count = self.kernels.shape[0]
        kernel_shape = self.kernels.shape[1:]

        # Get image matrix such that each row represents a patch of pixels in all batches that a single
        # kernel element was convolved with
        image_matrix = net.conversions.get_channels_wise_images_batch_patches_matrix(self.last_input, kernel_shape)

        # Change axes so that errors matrix is in order kernels, images, y, x
        errors_matrix = np.rollaxis(preactivation_error_gradients, 3, 0)

        # And now reshape errors matrix so that each column represents all errors for a single kernel across all images
        errors_matrix = errors_matrix.reshape(kernels_count, -1).T

        # Compute kernel gradients for all elements of all kernels
        # kernel_gradients_matrix is arranged so that a single column represents a flattened version of a single kernel
        kernel_gradients_matrix = np.dot(image_matrix, errors_matrix)

        # Reshape kernel matrix so that single kernel is in last dimension, then reshape whole matrix to
        # original kernels dimension and perform kernels update
        self.kernels -= learning_rate * kernel_gradients_matrix.T.reshape(self.kernels.shape) / self.last_input.shape[0]

    def _get_image_gradients(self, preactivation_error_gradients, kernels):

        kernels_patches_matrix = net.conversions.get_kernels_patches_matrix(kernels, self.last_input.shape[1:])

        # preactivation_error_gradients has shape images, y, x, output channels
        # Transform error gradients to shape images, output channels, y, x
        error_gradients = np.rollaxis(preactivation_error_gradients, 3, 1)

        # Now reshape to 2D matrix with each column standing for error from a single image
        error_gradients = error_gradients.reshape(self.last_input.shape[0], -1).T

        # Compute errors on each pixel
        pixel_errors_matrix = np.dot(kernels_patches_matrix, error_gradients)

        # Get transpose so that image count is in first dimesion again and reshape to 4D representing batch of 3D images
        image_gradients = pixel_errors_matrix.T.reshape(self.last_input.shape)

        return image_gradients


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

