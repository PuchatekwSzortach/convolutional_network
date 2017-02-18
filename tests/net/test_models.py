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
