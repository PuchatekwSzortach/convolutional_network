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

    def test_input_layer_non_3D_shape_fails(self):

        with pytest.raises(ValueError):

            net.layers.Input([2, 4])

    def test_input_layer_non_integer_shape_fails(self):

        with pytest.raises(ValueError):

            net.layers.Input([2.1, 4.2, 3.5])

    def test_build(self):

        input = net.layers.Input([2, 3, 4])
        input.build(input_shape=None)

        assert [2, 3, 4] == input.output_shape

    def test_forward(self):

        input = net.layers.Input([2, 3, 4])
        x = np.arange(4).reshape((2, 2))

        assert np.all(x == input.forward(x))


class TestFlattenLayer:
    """
    Tests for Flatten layer
    """

    def test_build_last_dimension_not_squeezed(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[1, 1, 4])

        assert [4] == flatten.output_shape

    def test_build_first_dimension_not_squeezed(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[3, 1, 1])

        assert [3] == flatten.output_shape

    def test_build_middle_dimension_not_squeezed(self):

        flatten = net.layers.Flatten()
        flatten.build(input_shape=[1, 5, 1])

        assert [5] == flatten.output_shape

    def test_forward_nothing_squeezed(self):

        flatten = net.layers.Flatten()
        x = np.arange(4).reshape((2, 2))

        expected = x
        actual = flatten.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_forward_with_squeeze(self):

        flatten = net.layers.Flatten()
        x = np.arange(8).reshape((2, 1, 2, 2))

        expected = np.squeeze(x)
        actual = flatten.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)

    def test_forward_batch_size_is_one(self):

        flatten = net.layers.Flatten()
        x = np.arange(4).reshape((1, 1, 2, 2))

        expected = np.arange(4).reshape((1, 2, 2))
        actual = flatten.forward(x)

        assert expected.shape == actual.shape
        assert np.all(expected == actual)
