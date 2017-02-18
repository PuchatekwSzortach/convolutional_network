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




# class TestFlattenLayer:
#     """
#     Tests for Flatten layer
#     """
#
#     def test_build_last_dimension_not_squeezed(self):
#
#         flatten = net.layers.Flatten()
#         flatten.build(input_shape=[1, 1, 4])
#
#         assert [4] == flatten.output_shape
#
#     def test_build_first_dimension_not_squeezed(self):
#
#         flatten = net.layers.Flatten()
#         flatten.build(input_shape=[3, 1, 1])
#
#         assert [3] == flatten.output_shape
#
#     def test_build_middle_dimension_not_squeezed(self):
#
#         flatten = net.layers.Flatten()
#         flatten.build(input_shape=[1, 5, 1])
#
#         assert [5] == flatten.output_shape
#
#     def test_forward_nothing_squeezed(self):
#
#         flatten = net.layers.Flatten()
#         x = np.arange(4).reshape((2, 2))
#
#         expected = x
#         actual = flatten.forward(x)
#
#         assert expected.shape == actual.shape
#         assert np.all(expected == actual)
#
#     def test_forward_with_squeeze(self):
#
#         flatten = net.layers.Flatten()
#         x = np.arange(8).reshape((2, 1, 2, 2))
#
#         expected = np.squeeze(x)
#         actual = flatten.forward(x)
#
#         assert expected.shape == actual.shape
#         assert np.all(expected == actual)
#
#     def test_forward_batch_size_is_one(self):
#
#         flatten = net.layers.Flatten()
#         x = np.arange(4).reshape((1, 1, 2, 2))
#
#         expected = np.arange(4).reshape((1, 2, 2))
#         actual = flatten.forward(x)
#
#         assert expected.shape == actual.shape
#         assert np.all(expected == actual)
