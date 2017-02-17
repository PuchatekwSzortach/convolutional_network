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

    pass
