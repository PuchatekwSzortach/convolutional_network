"""
Tests for net.models module
"""
import mock

import pytest
import numpy as np

import net.models
import net.layers


class TestModel:

    def test_constructor_check_layers_builds_called(self):

        layers = [mock.Mock() for _ in range(3)]

        net.models.Model(layers)

        assert layers[0].build.called
        assert layers[1].build.called
        assert layers[2].build.called

    def test_constructor_check_simple_model(self):

        input = net.layers.Input(sample_shape=(3, 1))
        flatten = net.layers.Flatten()

        net.models.Model([input, flatten])

        assert (None, 3, 1) == input.input_shape
        assert (None, 3, 1) == input.output_shape

        assert (None, 3, 1) == flatten.input_shape
        assert (None, 3) == flatten.output_shape

    def test_predict_simple(self):

        layers = [mock.Mock() for _ in range(3)]

        model = net.models.Model(layers)
        model.predict(x=1)

        assert layers[0].forward.called
        assert layers[1].forward.called
        assert layers[2].forward.called

    def test_predict_input_flatten_network(self):

        input = net.layers.Input(sample_shape=(3, 1))
        flatten = net.layers.Flatten()

        model = net.models.Model([input, flatten])

        x = np.arange(6).reshape((2, 3, 1))

        expected = x.reshape(2, 3)
        actual = model.predict(x)

        assert np.all(expected == actual)

    def test_predict_input_flatten_softmax_network(self):

        input = net.layers.Input(sample_shape=[2])
        flatten = net.layers.Flatten()
        softmax = net.layers.Softmax()

        model = net.models.Model([input, flatten, softmax])

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

        actual = model.predict(x)

        assert np.all(expected == actual)

    def test_predict_input_convolution_network(self):

        input = net.layers.Input(sample_shape=[3, 3, 1])
        convolution = net.layers.Convolution2D(filters=1, rows=2, columns=2)

        model = net.models.Model([input, convolution])

        x = np.array([
            [1, 0, 2],
            [2, 0, 1],
            [-1, 0, -1]
        ]).reshape((1, 3, 3, 1))

        kernel = np.array([
            [2, 3],
            [1, 2]
        ]).reshape(1, 2, 2, 1)

        # Overwrite kernels with known values
        convolution.kernels = kernel

        # Overwrite biases with known values
        convolution.biases = np.array([2])

        expected = np.array([
            [6, 10],
            [5, 3]
        ]).reshape((1, 2, 2, 1))

        actual = model.predict(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_predict_input_convolution_flatten_softmax_network(self):

        input = net.layers.Input(sample_shape=[3, 3, 2])
        convolution = net.layers.Convolution2D(filters=2, rows=3, columns=3)
        flatten = net.layers.Flatten()
        softmax = net.layers.Softmax()

        model = net.models.Model([input, convolution, flatten, softmax])

        first_channel = np.array([
            [1, 0, 2],
            [2, 0, 1],
            [-1, 0, -1]
        ])

        second_channel = np.array([
            [2, 1, 0],
            [1, 0, 2],
            [-4, 0, 1]
        ])

        x = np.dstack([first_channel, second_channel]).reshape((1, 3, 3, 2))

        first_kernel_first_channel = np.array([
            [2, 3, 0],
            [1, 2, 1],
            [0, -1, -1]
        ])

        first_kernel_second_channel = np.array([
            [1, 0, 2],
            [-1, 2, 0],
            [0, 1, 2]
        ])

        first_kernel = np.dstack([first_kernel_first_channel, first_kernel_second_channel])

        second_kernel_first_channel = np.array([
            [-1, -1, 0],
            [2, 0, 0],
            [1, 0, 4]
        ])

        second_kernel_second_channel = np.array([
            [2, 0, 0],
            [-1, -2, 0],
            [2, 3, 0]
        ])

        second_kernel = np.dstack([second_kernel_first_channel, second_kernel_second_channel])

        # Overwrite kernels with known values
        convolution.kernels = np.array([first_kernel, second_kernel])

        # Overwrite biases with known values
        convolution.biases = np.array([-2, 10])

        expected = np.array([0.982, 0.018]).reshape((1, 2))

        actual = model.predict(x)

        assert expected.shape == actual.shape
        assert np.allclose(expected, actual, atol=0.01)

    def test_get_loss_perfect_match(self):

        labels = np.array([
            [0, 1],
            [0, 1],
        ])

        predictions = np.array([
            [0, 1],
            [0, 1]
        ])

        epsilon = 1e-7
        expected = np.mean([epsilon, epsilon])

        actual = net.models.Model([]).get_loss(labels, predictions)

        assert np.allclose(expected, actual, atol=1e-4)

    def test_get_loss_perfect_mismatch(self):

        labels = np.array([
            [0, 1],
            [0, 1],
            [1, 0]
        ])

        predictions = np.array([
            [1, 0],
            [1, 0],
            [0, 1]
        ])

        epsilon = 1e-7
        expected = np.mean([-np.log(epsilon), -np.log(epsilon), -np.log(epsilon)])

        actual = net.models.Model([]).get_loss(labels, predictions)

        assert np.allclose(expected, actual, atol=1e-4)

    def test_get_loss_simple_values(self):

        labels = np.array([
            [0, 1],
            [0, 1],
            [1, 0]
        ])

        predictions = np.array([
            [0.2, 0.9],
            [0.8, 0.2],
            [0.3, 0.7]
        ])

        expected = np.mean([-np.log(0.9), -np.log(0.2), -np.log(0.3)])
        actual = net.models.Model([]).get_loss(labels, predictions)

        assert np.allclose(expected, actual, atol=1e-4)

    def test_get_accuracy(self):

        mock_layer = mock.Mock()
        mock_layer.forward.return_value = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ])

        model = net.models.Model([mock_layer])

        x = np.array([1])

        y = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        assert 0.25 == model.get_accuracy(x, y)

    def test_train_model(self):

        first_layer = net.layers.Convolution2D(filters=1, rows=2, columns=2)
        second_layer = net.layers.Convolution2D(filters=2, rows=1, columns=1)
        softmax = net.layers.Softmax()

        layers = [
            net.layers.Input(sample_shape=(2, 2, 2)),
            first_layer,
            second_layer,
            net.layers.Flatten(),
            softmax
        ]

        model = net.models.Model(layers)

        # Substitute kernels and biases with known values
        first_kernel_first_channel = np.array([
            [1, 2],
            [0, 2]
        ])

        first_kernel_second_channel = np.array([
            [1, 2],
            [-3, 0]
        ])

        first_layer.kernels = np.dstack(
            [first_kernel_first_channel, first_kernel_second_channel]).reshape(1, 2, 2, 2).astype(np.float32)

        first_layer.biases = np.array([-4], dtype=np.float32)

        second_layer.kernels = np.array([2, 1]).reshape(2, 1, 1, 1).astype(np.float32)

        second_layer.biases = np.array([-1, -2], dtype=np.float32)

        image_first_layer = np.array([
            [2, 2],
            [-1, 2]
        ])

        image_second_layer = np.array([
            [-1, 3],
            [2, 1]
        ])

        x = np.dstack([image_first_layer, image_second_layer]).reshape(1, 2, 2, 2)
        y = np.array([0, 1], dtype=np.float32).reshape(1, 2)

        model.train(x, y, learning_rate=0.5)

        # Assert convolution layers outputs
        assert np.all(np.array(5).reshape(1, 1, 1, 1) == first_layer.last_output)
        assert np.all(np.array([9, 3]).reshape(1, 1, 1, 2) == second_layer.last_output)

        # Assert softmax computations
        assert np.allclose([0.9975, 0.0025], softmax.last_output, atol=0.01)
        assert np.allclose([0.9975, -0.9975], softmax.get_output_layer_error_gradients(y), atol=0.01)

        # Assert second layer updates
        assert np.allclose(np.array([-0.49375, 3.49375]).reshape(2, 1, 1, 1), second_layer.kernels, atol=0.01)
        assert np.allclose(np.array([-1.49875, -1.50125]), second_layer.biases, atol=0.01)

        # Assert first layer updates
        expected_first_kernel_first_channel = np.array([
            [0.0025, 1.0025],
            [0.49875, 1.0025]
        ])

        expected_first_kernel_second_channel = np.array([
            [1.49875, 0.50375],
            [-3.9975, -0.49875]
        ])

        expected_first_kernel = np.dstack(
            [expected_first_kernel_first_channel, expected_first_kernel_second_channel]).reshape(1, 2, 2, 2)

        assert np.allclose(expected_first_kernel, first_layer.kernels, atol=0.01)
        assert np.allclose(np.array([-4.49925]), first_layer.biases, atol=0.01)
