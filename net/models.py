"""
Module with models definitions
"""


class Model:
    """
    Machine learning model, built from layers
    """

    def __init__(self, layers):
        """
        Constructor
        :param layers: list of Layers instances
        """

        self.layers = layers

        output_shape = None

        for layer in self.layers:

            layer.build(output_shape)
            output_shape = layer.output_shape

    def predict(self, x):

        for layer in self.layers:

            x = layer.forward(x)

        return x

