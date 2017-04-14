"""
Module with layers
"""

import itertools

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

        self.last_preactivation = None

    def build(self, input_shape):

        if len(input_shape) != 4:

            raise ValueError("Input shape {} isn't 4-dimensional".format(input_shape))

        self.input_shape = input_shape

        # Dimensions are [images, rows, columns, filters]
        self.output_shape = (None, input_shape[1] - self.nb_row + 1, input_shape[2] - self.nb_col + 1, self.nb_filter)

        input_channels = input_shape[-1]
        kernels_shape = (self.nb_filter, self.nb_row, self.nb_col, input_channels)

        scale = np.sqrt(2 / (self.nb_row * self.nb_col * input_channels))
        self.kernels = np.random.uniform(low=-scale, high=scale, size=kernels_shape)

        self.biases = np.zeros((self.nb_filter,), dtype=np.float32)

    def forward(self, input):

        preactivation = self._get_preactivation(input)

        # Apply ReLU activation
        return self._relu(preactivation)

    def _get_preactivation(self, input):

        preactivation = np.zeros(shape=(input.shape[0],) + self.output_shape[1:])

        # Kernels with kernels count dimension rolled to last dimension
        rolled_kernels = np.rollaxis(self.kernels, 0, 4)

        # For each image
        for sample_index, sample in enumerate(input):

            for row_index in range(0, input.shape[1] - self.nb_row + 1):

                for col_index in range(0, input.shape[2] - self.nb_col + 1):

                    # Select patch over which kernels will be applied
                    input_patch = sample[row_index: row_index + self.nb_row, col_index: col_index + self.nb_col, :]

                    # input patch is 3D, but we will convolve it with a series of 3D kernels, thus a 4D entity.
                    # Thus reshape input patch to have a 4th dimension. Numpy will then broadcast input patch
                    # along that dimension to match number of kernels
                    reshaped_input_patch = input_patch.reshape(input_patch.shape + (1,))

                    # Preactivation pixel value will a 1D array that holds results of convolutions over
                    # a selected patch with all kernels, one value for each kernel
                    preactivation_pixel_values = \
                        np.sum(reshaped_input_patch * rolled_kernels, axis=(0, 1, 2)) + self.biases

                    preactivation[sample_index, row_index, col_index, :] = preactivation_pixel_values

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
        return self._get_image_gradients(preactivation_error_gradients, old_kernels)

    def _update_bias(self, kernel_preactivation_error_gradients, kernel_index, learning_rate):

        # Sum contributions for each image
        bias_error_gradients_sums = np.sum(kernel_preactivation_error_gradients, axis=(1, 2))

        # And take mean across contributions to different images
        mean_bias_error_gradient = np.mean(bias_error_gradients_sums)

        self.biases[kernel_index] -= learning_rate * mean_bias_error_gradient

    def _update_kernels(self, preactivation_error_gradients, learning_rate):

        for kernel_index in range(len(self.kernels)):

            # Select gradients that were affected by current kernel
            # Dimensions are [image, rows, cols]
            kernel_preactivation_error_gradients = preactivation_error_gradients[:, :, :, kernel_index]

            self._update_bias(kernel_preactivation_error_gradients, kernel_index, learning_rate)
            self._update_kernel_weights(kernel_preactivation_error_gradients, kernel_index, learning_rate)

    def _update_kernel_weights(self, kernel_preactivation_error_gradients, kernel_index, learning_rate):

        for y in range(self.nb_row):

            input_row_start = y
            input_row_end = self.input_shape[1] - self.nb_row + y + 1

            for x in range(self.nb_col):

                input_column_start = x
                input_column_end = self.input_shape[2] - self.nb_col + x + 1

                inputs_patches = self.last_input[
                                 :, input_row_start:input_row_end, input_column_start:input_column_end, :]

                # Reshape kernel preactivation error gradients so they have a channel dimension
                # Numpy broadcasting will then take care of matching it with number of channels of inputs patches
                reshaped_kernel_preactivation_error_gradients = kernel_preactivation_error_gradients.reshape(
                    kernel_preactivation_error_gradients.shape + (1,))

                total_kernel_weight_gradient = np.sum(
                    reshaped_kernel_preactivation_error_gradients * inputs_patches, axis=(0, 1, 2))

                weight_index = (kernel_index, y, x)

                self.kernels[weight_index] -= learning_rate * total_kernel_weight_gradient / len(inputs_patches)

    def _get_image_gradients(self, preactivation_error_gradients, kernels):

        # Tensor for storing error gradients on input image
        image_gradients = np.zeros_like(self.last_input, dtype=np.float32)

        images_count = self.last_input.shape[0]
        rows_count = self.last_input.shape[1]
        columns_count = self.last_input.shape[2]

        # For each column
        for y in range(rows_count):

            # For each row
            for x in range(columns_count):

                # Select all kernel weights this pixel was convolved with
                # kernels ordering is: (self.nb_filter, self.nb_row, self.nb_col, input_channels)

                # Get start and end indices for part of kernel tensor that pixels I[y,x] was convolved with
                subkernel_y_start = max(0, y - self.last_preactivation.shape[1] + 1)
                subkernel_x_start = max(0, x - self.last_preactivation.shape[2] + 1)

                subkernel_y_end = min(kernels.shape[1], y + 1)
                subkernel_x_end = min(kernels.shape[2], x + 1)

                # Select indices of all elements in preactivation errors that were affected by examined image pixel
                y_slice_end = None if y - subkernel_y_end == -1 else y - subkernel_y_end
                y_slice = slice(y - subkernel_y_start, y_slice_end, -1)

                x_slice_end = None if x - subkernel_x_end == -1 else x - subkernel_x_end
                x_slice = slice(x - subkernel_x_start, x_slice_end, -1)

                # Error patch has dimensions images, y, x, output_channels
                error_patch = preactivation_error_gradients[:, y_slice, x_slice, :]

                # Select part of kernels that were convolved with image region currently computed over
                # subkernel has order kernels count, y, x, input channels
                subkernel = kernels[:, subkernel_y_start:subkernel_y_end, subkernel_x_start:subkernel_x_end, :]

                # error_patch has output channel dimension in last axis, but subkernel has it in first axis, since
                # error/output channels are equal/result of number of kernels. Roll kernels count
                # axis to end so that error patch and subkernel have output channels dimension in the same location
                # subkernel will now have order: y, x, input_channels, output_channels/kernels count
                subkernel = np.rollaxis(subkernel, 0, 4)

                # Now roll input channel dimension to end
                # subkernel will now have order: y, x, output_channels, input channels
                subkernel = np.rollaxis(subkernel, 2, 4)

                # error patch is a 4D tensor with dimensions image, y, x, output channels.
                # subkernel is a 4D tensor with dimensions y, x, output_channels, input_channels.
                # To make them compatible for element wise multiplication, we need error patch to include
                # input channel dimension and subkernel to include images dimension.

                # Repeat subkernel for each image
                # subkernel now has dimensions images, y, x, output_channels, input_channels
                subkernel = np.array([subkernel] * images_count)

                # Now add input channels dimension to error patch
                # error patch now has dimensions images, y, x, output_channels, input_channels
                # Numpy will broadcast error patch along last dimension to match size of subkernel
                error_patch = error_patch.reshape(error_patch.shape + (1, ))

                pixel_gradient = np.sum(error_patch * subkernel, axis=(1, 2, 3))

                # Dimensions are images, y, x, input channels
                pixel_index = (slice(None), y, x, slice(None))

                image_gradients[pixel_index] = pixel_gradient

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

