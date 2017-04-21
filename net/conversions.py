"""
Module with conversion functions for fast convolution operations
"""

import numpy as np


def get_image_patches_matrix(image, kernel_shape):
    """
    Convert a 3D image into a matrix of patches such that each row is a single patch
     that can be convolved with a kernel of provided shape
    :param image: 3D numpy array
    :param kernel_shape: kernel shape, 3rd dimension must be the same as that of image
    :return: 2D numpy array, rows represent image patches. Image patches are scanned in horizontal order
    with a step of 1
    """

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


def get_images_batch_patches_matrix(images, kernel_shape):
    """
    Convert a batch of 3D images into a single matrix of patches such that each row is a single patch
     that can be convolved with a kernel of provided shape
    :param images: 4D numpy array
    :param kernel_shape: kernel shape, 3rd dimension must be the same as last dimension of images array
    :return: 2D numpy array, rows represent image patches. Image patches are scanned in horizontal order
    with a step of 1. After one image ends, second image is scanned, and so on.
    """

    image_steps = images.shape[0]
    vertical_steps = images.shape[1] - kernel_shape[0] + 1
    horizontal_steps = images.shape[2] - kernel_shape[1] + 1

    shape = (image_steps * vertical_steps * horizontal_steps, np.product(kernel_shape))
    matrix = np.zeros(shape)

    for image_index in range(images.shape[0]):

        for row_index in range(images.shape[1] - kernel_shape[0] + 1):

            for column_index in range(images.shape[2] - kernel_shape[1] + 1):

                patch_index = (image_index, slice(row_index, row_index + kernel_shape[0]),
                               slice(column_index, column_index + kernel_shape[1]))

                image_patch = images[patch_index]

                matrix_index = (image_index * vertical_steps * horizontal_steps) + \
                               (row_index * horizontal_steps) + column_index

                matrix[matrix_index] = image_patch.flatten()

    return matrix


def get_channels_wise_image_patches_matrix(image, kernel_shape):
    """
    Convert a 3D image into a matrix of patches such that each row is a single patch
     that can be convolved with a kernel of provided shape. Each patch is over a single channel only
     and patches order is (rows, columns, channels) with channel index running fastest
    :param image: 3D numpy array
    :param kernel_shape: 2D kernel shape,
    :return: 2D numpy array, rows represent image patches. Image patches are scanned in
    with a step of 1 with fastest changing index being channels, then columns, then rows.
    """

    vertical_steps = image.shape[0] - kernel_shape[0] + 1
    horizontal_steps = image.shape[1] - kernel_shape[1] + 1

    # Row for each kernel element and each kernel element was convolved with vertical_steps * horizontal_steps pixels
    # from image
    shape = (np.prod(kernel_shape), vertical_steps * horizontal_steps)
    matrix = np.zeros(shape)

    for kernel_y in range(kernel_shape[0]):

        for kernel_x in range(kernel_shape[1]):

            for kernel_z in range(kernel_shape[2]):

                y_span = slice(kernel_y, image.shape[0] - kernel_shape[0] + kernel_y + 1)
                x_span = slice(kernel_x, image.shape[1] - kernel_shape[1] + kernel_x + 1)

                image_patch = image[y_span, x_span, kernel_z].flatten()

                row_index = (kernel_y * kernel_shape[1] * kernel_shape[2]) + (kernel_x * kernel_shape[2]) + kernel_z
                matrix[row_index] = image_patch

    return matrix