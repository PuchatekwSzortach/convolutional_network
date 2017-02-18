"""
Tests for net.layers module
"""

import pytest

import numpy as np

import net.layers


class TestInputLayer:
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


class TestFlattenLayer:
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


class TestSoftmaxLayer:
    """
    Tests for Softmax layer
    """

    def test_build_simple(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 10))

        assert (None, 10) == softmax.input_shape
        assert (None, 10) == softmax.output_shape

    def test_build_invalid_shape(self):

        softmax = net.layers.Softmax()

        with pytest.raises(ValueError):

            softmax.build(input_shape=(None, 20, 5))

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

    def test_forward_check_dimensions(self):

        softmax = net.layers.Softmax()
        softmax.build(input_shape=(None, 2))

        x = np.arange(16).reshape(2, 4, 2)

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
