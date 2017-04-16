"""
Module with conversion functions for fast convolution operations
"""

import numpy as np


def get_image_patches_matrix(image, kernel_shape):

    vertical_steps = image.shape[0] - kernel_shape[0] + 1
    horizontal_steps = image.shape[1] - kernel_shape[1] + 1

    shape = (vertical_steps * horizontal_steps, np.product(kernel_shape))
    image_matrix = np.zeros(shape)

    for row_index in range(image.shape[0] - kernel_shape[0] + 1):

        for column_index in range(image.shape[1] - kernel_shape[1] + 1):

            image_patch = image[
                          row_index: row_index + kernel_shape[0], column_index: column_index + kernel_shape[1]]

            index = (row_index * horizontal_steps) + column_index
            image_matrix[index] = image_patch.flatten()

    return image_matrix
