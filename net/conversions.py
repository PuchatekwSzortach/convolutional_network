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
    :param kernel_shape: 3D kernel shape,
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


def get_channels_wise_images_batch_patches_matrix(images, kernel_shape):
    """
    Convert a batch of 3D images into a matrix of patches such that each row is a single patch
     that can be convolved with a kernel of provided shape. Each patch is over a single channel only
     and patches order is (image, rows, columns, channels) with channel index running fastest
    :param images: a batch of 3D numpy arrays
    :param kernel_shape: 2D kernel shape,
    :return: 2D numpy array, rows represent image patches. Image patches are scanned in
    with a step of 1 with fastest changing index being channels, then columns, then rows, then images.
    """

    batch_size = images.shape[0]
    vertical_steps = images.shape[1] - kernel_shape[0] + 1
    horizontal_steps = images.shape[2] - kernel_shape[1] + 1

    # Row for each kernel element at each image.
    # Each kernel element was convolved with vertical_steps * horizontal_steps pixels over batch_size images
    shape = (np.prod(kernel_shape), batch_size * vertical_steps * horizontal_steps)
    matrix = np.zeros(shape)

    for kernel_y in range(kernel_shape[0]):

        for kernel_x in range(kernel_shape[1]):

            for kernel_z in range(kernel_shape[2]):

                y_span = slice(kernel_y, images.shape[1] - kernel_shape[0] + kernel_y + 1)
                x_span = slice(kernel_x, images.shape[2] - kernel_shape[1] + kernel_x + 1)

                patch_across_images = images[:, y_span, x_span, kernel_z].flatten()

                row_index = (kernel_y * kernel_shape[1] * kernel_shape[2]) + (kernel_x * kernel_shape[2]) + kernel_z
                matrix[row_index] = patch_across_images

    return matrix


def get_kernel_patches_matrix(kernel, image_shape):
    """
    Convert 3D kernel tensor into a 2D matrix such that each row of matrix represents
    kernel elements that a single element of image with image_shape was convolved with
    :param kernel: 3D numpy array
    :param image_shape: 3 elements tuple
    :return: 2D numpy array
    """

    rows_count = np.product(image_shape)

    errors_patch_y_shape = image_shape[0] - kernel.shape[0] + 1
    error_patch_x_shape = image_shape[1] - kernel.shape[1] + 1
    error_patch_shape = (errors_patch_y_shape, error_patch_x_shape)

    columns_count = np.product(error_patch_shape)

    matrix = np.zeros(shape=(rows_count, columns_count))

    for y in range(image_shape[0]):

        kernel_y_start = min(y, kernel.shape[0] - 1)
        kernel_y_end = max(-1, y - errors_patch_y_shape)
        kernel_y_range = range(kernel_y_start, kernel_y_end, -1)

        for x in range(image_shape[1]):

            kernel_x_start = min(x, kernel.shape[1] - 1)
            kernel_x_end = max(-1, x - error_patch_x_shape)
            kernel_x_range = range(kernel_x_start, kernel_x_end, -1)

            for z in range(image_shape[2]):

                row_index = (y * image_shape[1] * image_shape[2]) + (x * image_shape[2]) + z

                kernel_patch = np.zeros(error_patch_shape)

                for kernel_y in kernel_y_range:

                    for kernel_x in kernel_x_range:

                        # Get kernel element
                        kernel_patch[y - kernel_y, x - kernel_x] = kernel[kernel_y, kernel_x, z]

                matrix[row_index] = kernel_patch.flatten()

    return matrix