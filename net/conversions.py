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

    for row_index in range(images.shape[1] - kernel_shape[0] + 1):

        # Select at once rows from all images across all channels and columns
        images_rows_patches = images[:, row_index:row_index + kernel_shape[0]]

        for column_index in range(images.shape[2] - kernel_shape[1] + 1):

            # Now select columns from selected rows
            images_patches = images_rows_patches[:, :, column_index:column_index + kernel_shape[1]]

            # Select indices into appropriate rows for y and x indices of all images at once
            matrix_indices = (np.arange(image_steps) * vertical_steps * horizontal_steps) +\
                             (row_index * horizontal_steps) + column_index

            matrix[matrix_indices] = images_patches.reshape(image_steps, -1)

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

        y_span = slice(kernel_y, images.shape[1] - kernel_shape[0] + kernel_y + 1)

        # Select at once rows from all images across all channels and columns
        rows_patches_across_images = images[:, y_span]

        for kernel_x in range(kernel_shape[1]):

            x_span = slice(kernel_x, images.shape[2] - kernel_shape[1] + kernel_x + 1)

            # Now select columns from selected rows
            # patches_across_images has order images-y-x, z
            patches_across_images = rows_patches_across_images[:, :, x_span, :].reshape(-1, kernel_shape[2])

            row_index_start = (kernel_y * kernel_shape[1] * kernel_shape[2]) + (kernel_x * kernel_shape[2])
            row_index_end = row_index_start + kernel_shape[2]

            # Each row of matrix should be for a different channel, thus transpose patches_across_images
            # so their first dimension is channels
            matrix[row_index_start:row_index_end] = patches_across_images.T

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

    # Each pixel gets its own row
    rows_count = np.product(image_shape)

    kernels_count = kernels.shape[0]
    channels_count = image_shape[2]

    single_kernel_errors_patch_y_shape = image_shape[0] - kernels.shape[1] + 1
    single_kernel_errors_patch_x_shape = image_shape[1] - kernels.shape[2] + 1

    columns_count = kernels_count * single_kernel_errors_patch_y_shape * single_kernel_errors_patch_x_shape

    matrix = np.zeros(shape=(rows_count, columns_count))

    for y in range(image_shape[0]):

        kernel_y_start = min(y, kernels.shape[1] - 1)
        kernel_y_end = max(-1, y - single_kernel_errors_patch_y_shape)
        kernel_y_range = range(kernel_y_start, kernel_y_end, -1)

        # First select all appropriate rows and all channels
        # kernel_selection_rows has shape kernels, y, x, z
        kernels_selection_rows = kernels[:, kernel_y_range]

        for x in range(image_shape[1]):

            kernel_x_start = min(x, kernels.shape[2] - 1)
            kernel_x_end = max(-1, x - single_kernel_errors_patch_x_shape)
            kernel_x_range = range(kernel_x_start, kernel_x_end, -1)

            kernels_patch = np.zeros(
                (kernels.shape[0], single_kernel_errors_patch_y_shape,
                 single_kernel_errors_patch_x_shape, channels_count))

            # Then from these rows select all appropriate columns and all channels
            # kernel_selection has shape kernels, y, x, z
            kernels_selection = kernels_selection_rows[:, :, kernel_x_range]

            # Kernel patch is in order y, x, z
            kernels_patch[:, y - kernel_y_start:y - kernel_y_end, x - kernel_x_start:x - kernel_x_end] = \
                kernels_selection

            # Each matrix row corresponds to different channel, so move channel axis to beginning
            kernel_patch = np.rollaxis(kernels_patch, 3, 0)

            rows_index_start = (y * image_shape[1] * channels_count) + (x * channels_count)
            rows_index_end = rows_index_start + channels_count

            matrix[rows_index_start:rows_index_end] = kernel_patch.reshape(channels_count, -1)

    return matrix
