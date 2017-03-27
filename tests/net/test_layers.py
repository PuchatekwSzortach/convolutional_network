"""
Tests for net.layers module
"""

import pytest

import numpy as np

import net.layers


class TestInput:
    """
    Tests for Input layer class
    """

    def test_build_sample_shape_is_a_list(self):

        input = net.layers.Input(sample_shape=[2, 3, 4])
        input.build(input_shape=None)

        assert (None, 2, 3, 4) == input.input_shape
        assert (None, 2, 3, 4) == input.output_shape

    def test_build_sample_shape_is_a_tuple(self):

        input = net.layers.Input(sample_shape=(4, 3, 2))
        input.build(input_shape=None)

        assert (None, 4, 3, 2) == input.input_shape
        assert (None, 4, 3, 2) == input.output_shape

    def test_forward(self):

        input = net.layers.Input([2, 3, 4])
        x = np.arange(24).reshape((1, 2, 3, 4))

        assert np.all(x == input.forward(x))

    def test_forward_incompatible_shape(self):

        input = net.layers.Input([2, 3, 4])
        x = np.arange(48).reshape((1, 2, 3, 8))

        with pytest.raises(ValueError):

            input.forward(x)

    def test_train_forward_simple(self):

        input = net.layers.Input([4])
        x = np.array([1, 2, 3, 4]).reshape(1, 4)

        expected = x
        actual = input.train_forward(x)

        assert np.all(expected == actual)

    def test_train_backward_simple(self):

        input = net.layers.Input([4])

        gradients = np.array([1])

        assert input.train_backward(gradients) is None


class TestFlatten:
    """
    Tests for Flatten layer
    """

    def test_build_last_sample_dimension_not_squeezed(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[None, 1, 4])

        assert (None, 1, 4) == flatten.input_shape
        assert (None, 4) == flatten.output_shape

    def test_build_first_sample_dimension_not_squeezed(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[None, 5, 1])

        assert (None, 5, 1) == flatten.input_shape
        assert (None, 5) == flatten.output_shape

    def test_forward_nothing_to_squeeze(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[None, 3, 4])

        x = np.arange(24).reshape((2, 3, 4))

        expected = x
        actual = flatten.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_forward_invalid_input_shape(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[3, 4])
        x = np.arange(4).reshape((2, 2))

        with pytest.raises(ValueError):

            flatten.forward(x)

    def test_forward_with_squeeze(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[None, 1, 2, 2])
        x = np.arange(8).reshape((2, 1, 2, 2))

        expected = x.reshape((2, 2, 2))
        actual = flatten.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_forward_batch_size_is_one(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[None, 1, 2, 2])
        x = np.arange(4).reshape((1, 1, 2, 2))

        expected = np.arange(4).reshape((1, 2, 2))
        actual = flatten.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_train_backward_simple(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[None, 1, 1, 3])

        x = np.arange(6).reshape((2, 1, 1, 3))
        gradients = np.squeeze(2 * x)

        expected = 2 * x

        flatten.train_forward(x)
        actual = flatten.train_backward(gradients)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)


class TestSoftmax:
    """
    Tests for Softmax layer
    """

    def test_build_simple(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 10))

        assert (None, 10) == softmax.input_shape
        assert (None, 10) == softmax.output_shape

    def test_build_shape_more_than_2D(self):

        softmax = net.layers.Softmax()

        with pytest.raises(ValueError):

            softmax.build(input_shape=(None, 20, 5))

    def test_build_label_shape_less_than_two(self):

        softmax = net.layers.Softmax()

        with pytest.raises(ValueError):

            softmax.build(input_shape=(None, 1))

    def test_forward_simple(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 2))

        x = np.array(
            [
                [1, 2],
                [1, 4]
            ])

        expected = np.array(
            [
                [np.exp(1) / (np.exp(1) + np.exp(2)), np.exp(2) / (np.exp(1) + np.exp(2))],
                [np.exp(1) / (np.exp(1) + np.exp(4)), np.exp(4) / (np.exp(1) + np.exp(4))]
            ]
        )

        actual = softmax.forward(x)

        assert np.all(expected == actual)

    def test_forward_input_dimension_larger_than_2(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 2))

        x = np.arange(16).reshape(2, 4, 2)

        with pytest.raises(ValueError):

            softmax.forward(x)

    def test_forward_label_dimension_is_1(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 2))

        x = np.arange(10).reshape(10, 1)

        with pytest.raises(ValueError):

            softmax.forward(x)

    def test_forward_very_large_inputs(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 2))

        x = np.array(
            [
                [1, 2000],
                [5000, 4]
            ])

        expected = np.array(
            [
                [0, 1],
                [1, 0]
            ]
        )

        actual = softmax.forward(x)

        assert np.allclose(expected, actual)

    def test_get_output_layer_error_gradients_simple(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 2))

        x = np.array([
            [1, 2],
            [1, 4],
            [2, 3],
        ])

        y = np.array([
            [1, 0],
            [1, 0],
            [0, 1]
        ])

        expected = np.array([
            [0.269 - 1, 0.731],
            [0.047 - 1, 0.9523],
            [0.268, 0.731 - 1]
        ])

        softmax.train_forward(x)
        actual = softmax.get_output_layer_error_gradients(y)

        assert expected.shape == actual.shape
        assert np.allclose(expected, actual, atol=0.01)


class TestConvolution2D:
    """
    Tests for Convolution2D layer
    """

    def test_build_simple(self):

        convolution = net.layers.Convolution2D(nb_filter=3, nb_row=4, nb_col=5)
        convolution.build(input_shape=(None, 10, 10, 8))

        assert (None, 10, 10, 8) == convolution.input_shape
        assert (None, 7, 6, 3) == convolution.output_shape

        assert (3, 4, 5, 8) == convolution.kernels.shape
        assert (3,) == convolution.biases.shape

    def test_build_input_not_4D(self):

        convolution = net.layers.Convolution2D(nb_filter=3, nb_row=4, nb_col=5)

        with pytest.raises(ValueError):

            convolution.build(input_shape=(None, 10, 10))

    def test_forward_simple_one_input_channel_and_one_output_channel(self):

        convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
        convolution.build(input_shape=(None, 4, 4, 1))

        x = np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0]
            ]
        ).reshape((1, 4, 4, 1))

        kernel = np.array([
            [2, 3],
            [1, 2]
        ]).reshape(1, 2, 2, 1)

        # Overwrite kernels with known values
        convolution.kernels = kernel

        # Overwrite biases with known values
        convolution.biases = np.array([2])

        expected = np.array([
            [7, 6, 5],
            [3, 5, 9],
            [6, 5, 6]
        ]).reshape(1, 3, 3, 1)

        actual = convolution.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_forward_simple_one_input_channel_and_two_output_channels(self):

        convolution = net.layers.Convolution2D(nb_filter=2, nb_row=2, nb_col=2)
        convolution.build(input_shape=(None, 4, 4, 1))

        x = np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0]
            ]
        ).reshape((1, 4, 4, 1))

        first_kernel = np.array([
            [2, 3],
            [1, 2]
        ]).reshape(2, 2, 1)

        second_kernel = np.array([
            [-1, 2],
            [4, 0]
        ]).reshape(2, 2, 1)

        # Overwrite kernels with known values
        convolution.kernels = np.array([first_kernel, second_kernel])

        # Overwrite biases with known values
        convolution.biases = np.array([2, -2])

        expected_first_channel = np.array([
            [7, 6, 5],
            [3, 5, 9],
            [6, 5, 6]
        ])

        expected_second_channel = np.array([
            [0, 0, 2],
            [2, 0, 0],
            [0, 2, 4]
        ])

        expected = np.dstack([expected_first_channel, expected_second_channel]).reshape((1, 3, 3, 2))

        actual = convolution.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_train_forward_simple_one_input_channel_and_one_output_channel(self):

        convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
        convolution.build(input_shape=(None, 4, 4, 1))

        x = np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0]
            ]
        ).reshape((1, 4, 4, 1))

        kernel = np.array([
            [2, 3],
            [1, 2]
        ]).reshape(1, 2, 2, 1)

        # Overwrite kernels with known values
        convolution.kernels = kernel

        # Overwrite biases with known values
        convolution.biases = np.array([2])

        expected = np.array([
            [7, 6, 5],
            [3, 5, 9],
            [6, 5, 6]
        ]).reshape(1, 3, 3, 1)

        actual = convolution.train_forward(x)

        assert np.all(x == convolution.last_input)
        assert np.all(expected == actual)
        assert np.all(expected == convolution.last_output)

    def test_train_forward_two_2x2x1_images_one_2x2x1_kernel(self):

        convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
        convolution.build(input_shape=(None, 2, 2, 1))

        first_image = np.array([
            [2, 3],
            [-2, 0]
        ]).reshape(2, 2, 1)

        second_image = np.array([
            [-1, 1],
            [3, -1]
        ]).reshape(2, 2, 1)

        images = np.array([first_image, second_image])

        kernel = np.array([
            [2, 3],
            [1, 2]
        ]).reshape(1, 2, 2, 1)

        # Overwrite kernels with known values
        convolution.kernels = kernel

        # Overwrite biases with known values
        convolution.biases = np.array([-1])

        expected = np.array([10, 1]).reshape(2, 1, 1, 1)

        actual = convolution.forward(images)

        assert np.all(expected == actual)

    def test_train_forward_two_3x3x2_images_two_2x2x2_kernels(self):

        convolution = net.layers.Convolution2D(nb_filter=2, nb_row=2, nb_col=2)
        convolution.build(input_shape=(None, 3, 3, 2))

        first_image_first_channel = np.array([
            [-1, 2, 3],
            [1, 0, 1],
            [2, 2, 0]
        ])

        first_image_second_channel = np.array([
            [0, 4, 2],
            [-1, 1, 1],
            [0, 1, 0]
        ])

        first_image = np.dstack([first_image_first_channel, first_image_second_channel])

        second_image_first_channel = np.array([
            [3, -2, 0],
            [1, 1, 2],
            [0, 4, -2]
        ])

        second_image_second_channel = np.array([
            [1, -1, 2],
            [0, 3, 1],
            [1, 4, 0]
        ])

        second_image = np.dstack([second_image_first_channel, second_image_second_channel])

        images = np.array([first_image, second_image])

        first_kernel_first_channel = np.array([
            [-2, 0],
            [1, 1]
        ])

        first_kernel_second_channel = np.array([
            [0, -1],
            [2, 2]
        ])

        first_kernel = np.dstack([first_kernel_first_channel, first_kernel_second_channel])

        second_kernel_first_channel = np.array([
            [1, -2],
            [1, 0]
        ])

        second_kernel_second_channel = np.array([
            [3, 1],
            [0, 1]
        ])

        second_kernel = np.dstack([second_kernel_first_channel, second_kernel_second_channel])

        convolution.kernels = np.array([first_kernel, second_kernel])

        # Overwrite biases with known values
        convolution.biases = np.array([3, -1])

        expected_first_image_first_channel = np.array([
            [2, 2],
            [6, 6]
        ])

        expected_first_image_second_channel = np.array([
            [0, 10],
            [1, 3]
        ])

        expected_first_image = np.dstack([expected_first_image_first_channel, expected_first_image_second_channel])

        expected_second_image_first_channel = np.array([
            [6, 16],
            [12, 10]
        ])

        expected_second_image_second_channel = np.array([
            [12, 0],
            [5, 10]
        ])

        expected_second_image = np.dstack([expected_second_image_first_channel, expected_second_image_second_channel])

        expected_images = np.array([expected_first_image, expected_second_image])

        actual = convolution.forward(images)

        assert np.all(expected_images == actual)

    def test_train_backward_simple_one_input_channel_and_one_output_channel_single_sample_2x2_image(self):

        convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
        convolution.build(input_shape=(None, 2, 2, 1))

        image = np.array([
            [2, 3],
            [5, 1]
        ]).reshape((1, 2, 2, 1))

        kernel = np.array([
            [2, -2],
            [0, 1]
        ], dtype=np.float32).reshape(1, 2, 2, 1)

        # Overwrite kernels with known values
        convolution.kernels = kernel

        # Overwrite biases with known values
        convolution.biases = np.array([2], dtype=np.float32)

        expected_activation = np.array([1]).reshape(1, 1, 1, 1)

        actual_activation = convolution.train_forward(image)

        assert np.all(expected_activation == actual_activation)

        gradients = np.array([0.5]).reshape(1, 1, 1, 1)
        learning_rate = 1

        actual_image_gradients = convolution.train_backward(gradients, learning_rate)

        expected_biases = np.array([1.5])

        expected_kernels = np.array([
            [1, -3.5],
            [-2.5, 0.5]
        ]).reshape(1, 2, 2, 1)

        assert np.all(expected_biases == convolution.biases)
        assert np.all(expected_kernels == convolution.kernels)

        expected_image_gradients = np.array([
            [1, -1],
            [0, 0.5]
        ]).reshape(1, 2, 2, 1)

        assert np.all(expected_image_gradients == actual_image_gradients)
    #
    # def test_train_backward_simple_one_input_channel_and_one_output_channel_single_sample_3x3_image(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 3, 3, 1))
    #
    #     image = np.array([
    #         [2, 0, -1],
    #         [1, 1, 2],
    #         [3, -2, 0]
    #     ]).reshape((1, 3, 3, 1))
    #
    #     kernel = np.array([
    #         [2, -2],
    #         [0, 1]
    #     ], dtype=np.float32).reshape(1, 2, 2, 1)
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernel
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([4], dtype=np.float32)
    #
    #     expected_activation = np.array([
    #         [9, 8],
    #         [2, 2]
    #     ]).reshape(1, 2, 2, 1)
    #
    #     actual_activation = convolution.train_forward(image)
    #
    #     assert np.all(expected_activation == actual_activation)
    #
    #     gradients = np.array([
    #         [1, 2],
    #         [-1, -2]
    #     ]).reshape(1, 2, 2, 1)
    #
    #     learning_rate = 1
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([4])
    #
    #     expected_kernels = np.array([
    #         [3, 5],
    #         [-4, -6]
    #     ]).reshape(1, 2, 2, 1)
    #
    #     assert np.all(expected_biases == convolution.biases)
    #     assert np.all(expected_kernels == convolution.kernels)
    #
    # def test_train_backward_2x2x2_image_3_filters(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=3, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 2, 2, 2))
    #
    #     first_channel = np.array([
    #         [1, 3],
    #         [4, 0]
    #     ])
    #
    #     second_channel = np.array([
    #         [2, 0],
    #         [1, -2]
    #     ])
    #
    #     image = np.dstack([first_channel, second_channel]).reshape(1, 2, 2, 2)
    #
    #     first_kernel_first_channel = np.array([
    #         [2, -2],
    #         [0, 1]
    #     ], dtype=np.float32)
    #
    #     first_kernel_second_channel = np.array([
    #         [1, 0],
    #         [0, 1]
    #     ], dtype=np.float32)
    #
    #     first_kernel = np.dstack([first_kernel_first_channel, first_kernel_second_channel])
    #
    #     second_kernel_first_channel = np.array([
    #         [-1, 3],
    #         [0, 1]
    #     ], dtype=np.float32)
    #
    #     second_kernel_second_channel = np.array([
    #         [1, 1],
    #         [0, 0]
    #     ], dtype=np.float32)
    #
    #     second_kernel = np.dstack([second_kernel_first_channel, second_kernel_second_channel])
    #
    #     third_kernel_first_channel = np.array([
    #         [-2, 0],
    #         [1, 1]
    #     ], dtype=np.float32)
    #
    #     third_kernel_second_channel = np.array([
    #         [0, 1],
    #         [3, 0]
    #     ], dtype=np.float32)
    #
    #     third_kernel = np.dstack([third_kernel_first_channel, third_kernel_second_channel])
    #
    #     kernels = np.array([first_kernel, second_kernel, third_kernel])
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernels
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([2, -2, 3], dtype=np.float32)
    #
    #     expected_activation = np.array([
    #         [0, 8, 8],
    #     ]).reshape(1, 1, 1, 3)
    #
    #     actual_activation = convolution.train_forward(image)
    #
    #     assert np.all(expected_activation == actual_activation)
    #
    #     gradients = np.array([
    #         [1, 2, -4]
    #     ]).reshape(1, 1, 1, 3)
    #
    #     learning_rate = 1
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([2, -4, 7])
    #
    #     assert np.all(expected_biases == convolution.biases)
    #
    #     first_kernel_first_channel_expected = np.array([
    #         [2, -2],
    #         [0, 1]
    #     ])
    #
    #     first_kernel_second_channel_expected = np.array([
    #         [1, 0],
    #         [0, 1]
    #     ])
    #
    #     first_kernel_expected = np.dstack([first_kernel_first_channel_expected, first_kernel_second_channel_expected])
    #
    #     second_kernel_first_channel_expected = np.array([
    #         [-3, -3],
    #         [-8, 1]
    #     ])
    #
    #     second_kernel_second_channel_expected = np.array([
    #         [-3, 1],
    #         [-2, 4]
    #     ])
    #
    #     second_kernel_expected = np.dstack([second_kernel_first_channel_expected, second_kernel_second_channel_expected])
    #
    #     third_kernel_first_channel_expected = np.array([
    #         [2, 12],
    #         [17, 1]
    #     ])
    #
    #     third_kernel_second_channel_expected = np.array([
    #         [8, 1],
    #         [7, -8]
    #     ])
    #
    #     third_kernel_expected = np.dstack(
    #         [third_kernel_first_channel_expected, third_kernel_second_channel_expected])
    #
    #     expected_kernels = np.array([first_kernel_expected, second_kernel_expected, third_kernel_expected])
    #
    #     assert np.all(expected_kernels == convolution.kernels)
    #
    # def test_train_backward_one_3x3x2_image_2_kernels_2x2x2(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=2, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 3, 3, 2))
    #
    #     first_channel = np.array([
    #         [1, 2, 0],
    #         [-1, 0, 3],
    #         [2, 2, 0]
    #     ])
    #
    #     second_channel = np.array([
    #         [0, -2, 1],
    #         [3, 1, 1],
    #         [1, 2, 0]
    #     ])
    #
    #     image = np.dstack([first_channel, second_channel]).reshape(1, 3, 3, 2)
    #
    #     first_kernel_first_channel = np.array([
    #         [2, 0],
    #         [1, 1]
    #     ])
    #
    #     first_kernel_second_channel = np.array([
    #         [1, -1],
    #         [2, 0]
    #     ])
    #
    #     first_kernel = np.dstack([first_kernel_first_channel, first_kernel_second_channel])
    #
    #     second_kernel_first_channel = np.array([
    #         [-1, 3],
    #         [1, 0]
    #     ])
    #
    #     second_kernel_second_channel = np.array([
    #         [2, -1],
    #         [1, 1]
    #     ])
    #
    #     second_kernel = np.dstack([second_kernel_first_channel, second_kernel_second_channel])
    #
    #     kernels = np.array([first_kernel, second_kernel])
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernels
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([-1, 4], dtype=np.float32)
    #
    #     expected_activation_first_channel = np.array([
    #         [8, 5],
    #         [5, 5]
    #     ])
    #
    #     expected_activation_second_channel = np.array([
    #         [14, 0],
    #         [15, 18]
    #     ])
    #
    #     expected_activation = np.dstack([expected_activation_first_channel, expected_activation_second_channel])\
    #         .reshape(1, 2, 2, 2)
    #
    #     actual_activation = convolution.train_forward(image)
    #
    #     assert np.all(expected_activation == actual_activation)
    #
    #     first_channel_gradients = np.array([
    #         [-1, 2],
    #         [1, 3]
    #     ])
    #
    #     second_channel_gradients = np.array([
    #         [1, 0],
    #         [2, -4]
    #     ])
    #
    #     gradients = np.dstack([first_channel_gradients, second_channel_gradients]).reshape(1, 2, 2, 2)
    #
    #     learning_rate = 1
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([-6, 5])
    #
    #     assert np.all(expected_biases == convolution.biases)
    #
    #     first_kernel_first_channel_expected = np.array([
    #         [0, -7],
    #         [-8, -7]
    #     ])
    #
    #     first_kernel_second_channel_expected = np.array([
    #         [-1, -9],
    #         [-4, -3]
    #     ])
    #
    #     first_kernel_expected = np.dstack([first_kernel_first_channel_expected, first_kernel_second_channel_expected])
    #
    #     second_kernel_first_channel_expected = np.array([
    #         [0, 13],
    #         [6, -4]
    #     ])
    #
    #     second_kernel_second_channel_expected = np.array([
    #         [0, 3],
    #         [4, -4]
    #     ])
    #
    #     second_kernel_expected = np.dstack([second_kernel_first_channel_expected, second_kernel_second_channel_expected])
    #
    #     expected_kernels = np.array([first_kernel_expected, second_kernel_expected])
    #
    #     assert np.all(expected_kernels == convolution.kernels)
    #
    # def test_train_backward_two_2x2x1_images_one_2x2x1_kernel(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 2, 2, 1))
    #
    #     first_image = np.array([
    #         [1, -2],
    #         [4, 3]
    #     ]).reshape(2, 2, 1)
    #
    #     second_image = np.array([
    #         [2, 1],
    #         [3, 2]
    #     ]).reshape(2, 2, 1)
    #
    #     images = np.array([first_image, second_image]).astype(np.float32)
    #
    #     kernel = np.array([
    #         [1, 2],
    #         [4, -1]
    #     ]).reshape(1, 2, 2, 1).astype(np.float32)
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernel
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([2], dtype=np.float32)
    #
    #     expected_activation = np.array([12, 16]).reshape(2, 1, 1, 1)
    #
    #     actual_activation = convolution.train_forward(images)
    #
    #     assert np.all(expected_activation == actual_activation)
    #
    #     gradients = np.array([1, 2]).reshape(2, 1, 1, 1).astype(np.float32)
    #     learning_rate = 1
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([0.5])
    #
    #     assert np.all(expected_biases == convolution.biases)
    #
    #     expected_kernels = np.array([
    #         [-1.5, 2],
    #         [-1, -4.5]
    #     ]).reshape(1, 2, 2, 1)
    #
    #     assert np.all(expected_kernels == convolution.kernels)
    #
    # def test_train_backward_two_2x2x2_images_one_2x2x2_kernel(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 2, 2, 2))
    #
    #     first_image_first_channel = np.array([
    #         [1, 0],
    #         [-2, 3]
    #     ])
    #
    #     first_image_second_channel = np.array([
    #         [1, 1],
    #         [0, 2]
    #     ])
    #
    #     first_image = np.dstack([first_image_first_channel, first_image_second_channel])
    #
    #     second_image_first_channel = np.array([
    #         [3, -1],
    #         [0, 1]
    #     ])
    #
    #     second_image_second_channel = np.array([
    #         [0, 3],
    #         [-1,  2]
    #     ])
    #
    #     second_image = np.dstack([second_image_first_channel, second_image_second_channel])
    #
    #     images = np.array([first_image, second_image])
    #
    #     kernel_first_channel = np.array([
    #         [3, 2],
    #         [-1, 0]
    #     ])
    #
    #     kernel_second_channel = np.array([
    #         [0, -2],
    #         [1, 0]
    #     ])
    #
    #     kernel = np.dstack([kernel_first_channel, kernel_second_channel]).reshape(1, 2, 2, 2)
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernel
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([3], dtype=np.float32)
    #
    #     expected_first_activation = np.array([6]).reshape(1, 1, 1)
    #     expected_second_activation = np.array([3]).reshape(1, 1, 1)
    #
    #     expected_activation = np.array([expected_first_activation, expected_second_activation])
    #
    #     actual_activation = convolution.train_forward(images)
    #
    #     assert np.all(expected_activation == actual_activation)
    #
    #     gradients = np.array([2, 4]).reshape(2, 1, 1, 1).astype(np.float32)
    #     learning_rate = 1
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([0])
    #
    #     assert np.all(expected_biases == convolution.biases)
    #
    #     expected_kernels_first_channel = np.array([
    #         [-4, 4],
    #         [1, -5]
    #     ])
    #
    #     expected_kernels_second_channel = np.array([
    #         [-1, -9],
    #         [3, -6]
    #     ])
    #
    #     expected_kernels = np.dstack(
    #         [expected_kernels_first_channel, expected_kernels_second_channel]).reshape(1, 2, 2, 2)
    #
    #     assert np.all(expected_kernels == convolution.kernels)
    #
    # def test_train_backward_two_3x3x2_images_one_2x2x2_kernel(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=1, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 3, 3, 2))
    #
    #     first_image_first_channel = np.array([
    #         [1, 0, 2],
    #         [1, -2, 3],
    #         [-2, 0, 3]
    #     ])
    #
    #     first_image_second_channel = np.array([
    #         [1, 1, -4],
    #         [3, 0, 2],
    #         [-1, -1, 2]
    #     ])
    #
    #     first_image = np.dstack([first_image_first_channel, first_image_second_channel])
    #
    #     second_image_first_channel = np.array([
    #         [-1, 2, 0],
    #         [1, 1, 1],
    #         [0, 0, 3]
    #     ])
    #
    #     second_image_second_channel = np.array([
    #         [2, 0, -1],
    #         [1, 1, 0],
    #         [-1, -1, 1]
    #     ])
    #
    #     second_image = np.dstack([second_image_first_channel, second_image_second_channel])
    #
    #     images = np.array([first_image, second_image])
    #
    #     kernel_first_channel = np.array([
    #         [1, 2],
    #         [-1, 0]
    #     ])
    #
    #     kernel_second_channel = np.array([
    #         [0, 6],
    #         [-3, 0]
    #     ])
    #
    #     kernel = np.dstack([kernel_first_channel, kernel_second_channel]).reshape(1, 2, 2, 2)
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernel
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([2], dtype=np.float32)
    #
    #     expected_first_activation = np.array([
    #         [0, 0],
    #         [4, 21]
    #     ]).reshape(2, 2, 1)
    #
    #     expected_second_activation = np.array([
    #         [1, 0],
    #         [14, 8]
    #     ]).reshape(2, 2, 1)
    #
    #     expected_activation = np.array([expected_first_activation, expected_second_activation])
    #
    #     actual_activation = convolution.train_forward(images)
    #
    #     assert np.all(expected_activation == actual_activation)
    #
    #     first_image_gradients = np.array([
    #         [2, -2],
    #         [1, 1]
    #     ]).reshape(2, 2, 1)
    #
    #     second_image_gradients = np.array([
    #         [3, 1],
    #         [-1, 2]
    #     ]).reshape(2, 2, 1)
    #
    #     gradients = np.array([first_image_gradients, second_image_gradients])
    #
    #     learning_rate = 2
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([-4])
    #
    #     assert np.all(expected_biases == convolution.biases)
    #
    #     expected_kernels_first_channel = np.array([
    #         [4, -6],
    #         [-2, -12]
    #     ])
    #
    #     expected_kernels_second_channel = np.array([
    #         [-10, 5],
    #         [-3, -7]
    #     ])
    #
    #     expected_kernels = np.dstack(
    #         [expected_kernels_first_channel, expected_kernels_second_channel]).reshape(1, 2, 2, 2)
    #
    #     assert np.all(expected_kernels == convolution.kernels)
    #
    # def test_train_backward_two_3x3x2_images_two_2x2x2_kernels(self):
    #
    #     convolution = net.layers.Convolution2D(nb_filter=2, nb_row=2, nb_col=2)
    #     convolution.build(input_shape=(None, 3, 3, 2))
    #
    #     first_image_first_channel = np.array([
    #         [3, 0, 1],
    #         [1, 1, 0],
    #         [-1, -1, 0]
    #     ])
    #
    #     first_image_second_channel = np.array([
    #         [2, 0, -1],
    #         [1, 1, 2],
    #         [0, -2, 1]
    #     ])
    #
    #     first_image = np.dstack([first_image_first_channel, first_image_second_channel])
    #
    #     second_image_first_channel = np.array([
    #         [0, 1, 1],
    #         [0, 1, 0],
    #         [-3, -1, 1]
    #     ])
    #
    #     second_image_second_channel = np.array([
    #         [0, 2, -2],
    #         [4, 0, 1],
    #         [-1, -1, 0]
    #     ])
    #
    #     second_image = np.dstack([second_image_first_channel, second_image_second_channel])
    #
    #     images = np.array([first_image, second_image])
    #
    #     first_kernel_first_channel = np.array([
    #         [1, 1],
    #         [0, -2]
    #     ])
    #
    #     first_kernel_second_channel = np.array([
    #         [0, 1],
    #         [2, 0]
    #     ])
    #
    #     first_kernel = np.dstack([first_kernel_first_channel, first_kernel_second_channel])
    #
    #     second_kernel_first_channel = np.array([
    #         [-1, 0],
    #         [1, 0]
    #     ])
    #
    #     second_kernel_second_channel = np.array([
    #         [2, -4],
    #         [3, 0]
    #     ])
    #
    #     second_kernel = np.dstack([second_kernel_first_channel, second_kernel_second_channel])
    #
    #     kernels = np.array([first_kernel, second_kernel])
    #
    #     # Overwrite kernels with known values
    #     convolution.kernels = kernels
    #
    #     # Overwrite biases with known values
    #     convolution.biases = np.array([1, 4], dtype=np.float32)
    #
    #     expected_first_activation_first_channel = np.array([
    #         [4, 3],
    #         [6, 0]
    #     ]).reshape(2, 2, 1)
    #
    #     expected_first_activation_second_channel = np.array([
    #         [9, 12],
    #         [0, 0]
    #     ]).reshape(2, 2, 1)
    #
    #     expected_first_activation = np.dstack(
    #         [expected_first_activation_first_channel, expected_first_activation_second_channel])
    #
    #     expected_second_activation_first_channel = np.array([
    #         [10, 1],
    #         [2, 0]
    #     ]).reshape(2, 2, 1)
    #
    #     expected_second_activation_second_channel = np.array([
    #         [8, 16],
    #         [6, 0]
    #     ]).reshape(2, 2, 1)
    #
    #     expected_second_activation = np.dstack(
    #         [expected_second_activation_first_channel, expected_second_activation_second_channel])
    #
    #     expected_activations = np.array([expected_first_activation, expected_second_activation])
    #
    #     actual_activations = convolution.train_forward(images)
    #
    #     assert np.all(expected_activations == actual_activations)
    #
    #     first_image_gradients_first_channel = np.array([
    #         [1, -1],
    #         [0, 3]
    #     ]).reshape(2, 2, 1)
    #
    #     first_image_gradients_second_channel = np.array([
    #         [3, 2],
    #         [0, 1]
    #     ]).reshape(2, 2, 1)
    #
    #     first_image_gradients = np.dstack(
    #         [first_image_gradients_first_channel, first_image_gradients_second_channel])
    #
    #     second_image_gradients_first_channel = np.array([
    #         [2, 1],
    #         [0, 2]
    #     ]).reshape(2, 2, 1)
    #
    #     second_image_gradients_second_channel = np.array([
    #         [0, 3],
    #         [1, 1]
    #     ]).reshape(2, 2, 1)
    #
    #     second_image_gradients = np.dstack(
    #         [second_image_gradients_first_channel, second_image_gradients_second_channel])
    #
    #     gradients = np.array([first_image_gradients, second_image_gradients])
    #
    #     learning_rate = 2
    #
    #     convolution.train_backward(gradients, learning_rate)
    #
    #     expected_biases = np.array([-2, -5])
    #
    #     assert np.all(expected_biases == convolution.biases)
    #
    #     expected_first_kernel_first_channel = np.array([
    #         [-3, -1],
    #         [-1, -5]
    #     ])
    #
    #     expected_first_kernel_second_channel = np.array([
    #         [-4, -2],
    #         [-6, 0]
    #     ])
    #
    #     expected_first_kernel = np.dstack([expected_first_kernel_first_channel, expected_first_kernel_second_channel])
    #
    #     expected_second_kernel_first_channel = np.array([
    #         [-13, -6],
    #         [-4, -2]
    #     ])
    #
    #     expected_second_kernel_second_channel = np.array([
    #         [-14, 4],
    #         [-1, -9]
    #     ])
    #
    #     expected_second_kernel = np.dstack([expected_second_kernel_first_channel, expected_second_kernel_second_channel])
    #
    #     expected_kernels = np.array([expected_first_kernel, expected_second_kernel])
    #
    #     assert np.all(expected_kernels == convolution.kernels)

