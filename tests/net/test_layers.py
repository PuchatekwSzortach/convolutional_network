"""
Tests for net.layers module
"""

import pytest

import net.layers


def test_input_layer_non_3D_shape_fails():

    with pytest.raises(ValueError):

        net.layers.Input([2, 4])


def test_input_layer_non_integer_shape_fails():

    with pytest.raises(ValueError):

        net.layers.Input([2.1, 4.2, 3.5])
