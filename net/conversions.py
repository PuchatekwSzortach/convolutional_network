"""
Module with conversion functions for fast convolution operations
"""

import numpy as np


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


def get_kernels_patches_matrix(kernels, image_shape):
    """
    Convert 4D kernels tensor into a 2D matrix such that each row of matrix represents
    kernel elements that a single element of image with image_shape was convolved with.
    Kernels are spread out in each row in order, first elements from first kernel, then second, etc.
    :param kernels: 4D numpy array
    :param image_shape: 3 elements tuple
    :return: 2D numpy array
    """

    rows_count = np.product(image_shape)

    kernels_count = kernels.shape[0]
    single_kernel_errors_patch_y_shape = image_shape[0] - kernels.shape[1] + 1
    single_kernel_errors_patch_x_shape = image_shape[1] - kernels.shape[2] + 1
    single_kernel_errors_patch_shape = (single_kernel_errors_patch_y_shape, single_kernel_errors_patch_x_shape)

    columns_count = kernels_count * single_kernel_errors_patch_y_shape * single_kernel_errors_patch_x_shape

    matrix = np.zeros(shape=(rows_count, columns_count))

    for kernel_index in range(kernels_count):

        kernel_offset_start = kernel_index * single_kernel_errors_patch_y_shape * single_kernel_errors_patch_x_shape
        kernel_offset_end = (kernel_index + 1) * single_kernel_errors_patch_y_shape * single_kernel_errors_patch_x_shape

        for y in range(image_shape[0]):

            kernel_y_start = min(y, kernels.shape[1] - 1)
            kernel_y_end = max(-1, y - single_kernel_errors_patch_y_shape)
            kernel_y_range = range(kernel_y_start, kernel_y_end, -1)

            for x in range(image_shape[1]):

                kernel_x_start = min(x, kernels.shape[2] - 1)
                kernel_x_end = max(-1, x - single_kernel_errors_patch_x_shape)
                kernel_x_range = range(kernel_x_start, kernel_x_end, -1)

                for z in range(image_shape[2]):

                    row_index = (y * image_shape[1] * image_shape[2]) + (x * image_shape[2]) + z

                    kernel_patch = np.zeros(single_kernel_errors_patch_shape)

                    # First select all appropriate rows
                    kernel_selection_rows = kernels[kernel_index, kernel_y_range, :, z]

                    # Then from these rows select all appropriate columns
                    kernel_selection = kernel_selection_rows[:, kernel_x_range]

                    kernel_patch[y - kernel_y_start:y - kernel_y_end, x - kernel_x_start:x - kernel_x_end] = \
                        kernel_selection

                    matrix[row_index, kernel_offset_start:kernel_offset_end] = kernel_patch.flatten()

    return matrix
