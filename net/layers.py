"""
Module with layers
"""


class Input:
    """
    Input layer
    """

    def __init__(self, shape):
        """
        Constructor
        :param shape: 3 elements tuple (height, width, channels)
        """

        if len(shape) != 3:

            raise ValueError("Shape must have 3 elements, but {} were provided".format(len(shape)))

        are_dimensions_integers = [isinstance(x, int) for x in shape]

        if not all(are_dimensions_integers):

            raise ValueError("Not all elements of shape {} are integers".format(shape))

        self.shape = shape