"""
Module with layers
"""


class Layer:

    def __init__(self):

        self.output_shape = None
        self.activation = None

    def build(self):

        raise NotImplementedError()

    def forward(self, x):

        raise NotImplementedError()

    def train_forward(self):

        raise NotImplementedError()

    def train_backward(self):

        raise NotImplementedError()


class Input(Layer):
    """
    Input layer
    """

    def __init__(self, shape):
        """
        Constructor
        :param shape: 3 elements tuple (height, width, channels)
        """

        super().__init__()

        if len(shape) != 3:

            raise ValueError("Shape must have 3 elements, but {} were provided".format(len(shape)))

        are_dimensions_integers = [isinstance(x, int) for x in shape]

        if not all(are_dimensions_integers):

            raise ValueError("Not all elements of shape {} are integers".format(shape))

        self.output_shape = shape

    def build(self):

        pass

    def forward(self, x):

        return x