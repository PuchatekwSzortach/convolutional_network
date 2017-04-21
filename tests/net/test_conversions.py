"""
Test for net.conversions module
"""

import numpy as np

import net.conversions


def test_get_image_patches_matrix_2_2_1_image():

    image = np.arange(4).reshape(2, 2, 1)
    kernel_shape = (2, 2, 1)

    expected = np.array([0, 1, 2, 3]).reshape(1, 4)
    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_image_patches_matrix_2_2_2_image():

    image = np.arange(8).reshape(2, 2, 2)
    kernel_shape = (2, 2, 2)

    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(1, 8)
    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_image_patches_matrix_3_3_1_image():

    image = np.arange(9).reshape(3, 3, 1)
    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 8]
    ])

    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_image_patches_matrix_3_3_2_image():

    image = np.arange(18).reshape(3, 3, 2)
    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0, 1, 2, 3, 6, 7, 8, 9],
        [2, 3, 4, 5, 8, 9, 10, 11],
        [6, 7, 8, 9, 12, 13, 14, 15],
        [8, 9, 10, 11, 14, 15, 16, 17]
    ])

    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_image_patches_matrix_4_2_1_image():

    image = np.arange(8).reshape(4, 2, 1)
    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7]
    ])

    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_image_patches_matrix_2_4_1_image():

    image = np.arange(8).reshape(2, 4, 1)
    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 4, 5],
        [1, 2, 5, 6],
        [2, 3, 6, 7]
    ])

    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_image_patches_matrix_4x4x1_image():

    image = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])

    kernel_shape = (2, 2, 1)

    expected = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ])

    actual = net.conversions.get_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_images_batch_patches_matrix_single_2x2x1_image():

    images = np.arange(4).reshape(1, 2, 2, 1)
    kernel_shape = (2, 2, 1)

    expected = np.array([0, 1, 2, 3]).reshape(1, 4)
    actual = net.conversions.get_images_batch_patches_matrix(images, kernel_shape)

    assert np.all(expected == actual)


def test_get_images_batch_patches_matrix_two_2x2x1_images():

    first_image = np.arange(4).reshape(2, 2, 1)
    second_image = np.arange(10, 14).reshape(2, 2, 1)

    images = np.array([first_image, second_image])
    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13]
    ])

    actual = net.conversions.get_images_batch_patches_matrix(images, kernel_shape)

    assert np.all(expected == actual)


def test_get_images_batch_patches_matrix_two_2x2x2_images():

    first_image = np.arange(8).reshape(2, 2, 2)
    second_image = np.arange(10, 18).reshape(2, 2, 2)

    images = np.array([first_image, second_image])
    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [10, 11, 12, 13, 14, 15, 16, 17],
    ])

    actual = net.conversions.get_images_batch_patches_matrix(images, kernel_shape)

    assert np.all(expected == actual)


def test_get_images_batch_patches_matrix_two_3x3x2_images_2x2x2_kernel():

    first_image = np.arange(18).reshape(3, 3, 2)
    second_image = np.arange(20, 38).reshape(3, 3, 2)

    images = np.array([first_image, second_image])
    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0, 1, 2, 3, 6, 7, 8, 9],
        [2, 3, 4, 5, 8, 9, 10, 11],
        [6, 7, 8, 9, 12, 13, 14, 15],
        [8, 9, 10, 11, 14, 15, 16, 17],
        [20, 21, 22, 23, 26, 27, 28, 29],
        [22, 23, 24, 25, 28, 29, 30, 31],
        [26, 27, 28, 29, 32, 33, 34, 35],
        [28, 29, 30, 31, 34, 35, 36, 37]
    ])

    actual = net.conversions.get_images_batch_patches_matrix(images, kernel_shape)

    assert np.all(expected == actual)


def test_get_images_batch_patches_matrix_two_4x2x1_images_2x2x1_kernel():

    first_image = np.arange(8).reshape(4, 2, 1)
    second_image = np.arange(10, 18).reshape(4, 2, 1)

    images = np.array([first_image, second_image])
    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [10, 11, 12, 13],
        [12, 13, 14, 15],
        [14, 15, 16, 17]
    ])

    actual = net.conversions.get_images_batch_patches_matrix(images, kernel_shape)

    assert np.all(expected == actual)


def test_get_images_batch_patches_matrix_two_2x4x1_images_2x2x1_kernel():

    first_image = np.arange(8).reshape(2, 4, 1)
    second_image = np.arange(10, 18).reshape(2, 4, 1)

    images = np.array([first_image, second_image])
    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 4, 5],
        [1, 2, 5, 6],
        [2, 3, 6, 7],
        [10, 11, 14, 15],
        [11, 12, 15, 16],
        [12, 13, 16, 17]
    ])

    actual = net.conversions.get_images_batch_patches_matrix(images, kernel_shape)

    assert np.all(expected == actual)


def test_get_channels_wise_image_patches_matrix_2x2x1_image_2x2x1_kernel():

    image = np.arange(4).reshape(2, 2, 1)

    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0],
        [1],
        [2],
        [3]
    ])

    actual = net.conversions.get_channels_wise_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_channels_wise_image_patches_matrix_3x3x1_image_2x2x1_kernel():

    image = np.arange(9).reshape(3, 3, 1)

    kernel_shape = (2, 2, 1)

    expected = np.array([
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 8]
    ])

    actual = net.conversions.get_channels_wise_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_channels_wise_image_patches_matrix_2x2x2_image_2x2x2_kernel():

    image = np.arange(8).reshape(2, 2, 2)

    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7]
    ])

    actual = net.conversions.get_channels_wise_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_channels_wise_image_patches_matrix_3x3x2_image_2x2x2_kernel():

    image = np.arange(18).reshape(3, 3, 2)

    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0, 2, 6, 8],
        [1, 3, 7, 9],
        [2, 4, 8, 10],
        [3, 5, 9, 11],
        [6, 8, 12, 14],
        [7, 9, 13, 15],
        [8, 10, 14, 16],
        [9, 11, 15, 17]
    ])

    actual = net.conversions.get_channels_wise_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_channels_wise_image_patches_matrix_2x4x2_image_2x2x2_kernel():

    image = np.arange(16).reshape(2, 4, 2)
    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0, 2, 4],
        [1, 3, 5],
        [2, 4, 6],
        [3, 5, 7],
        [8, 10, 12],
        [9, 11, 13],
        [10, 12, 14],
        [11, 13, 15]
    ])

    actual = net.conversions.get_channels_wise_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)


def test_get_channels_wise_image_patches_matrix_4x2x2_image_2x2x2_kernel():

    image = np.arange(16).reshape(4, 2, 2)
    kernel_shape = (2, 2, 2)

    expected = np.array([
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
        [4, 8, 12],
        [5, 9, 13],
        [6, 10, 14],
        [7, 11, 15]
    ])

    actual = net.conversions.get_channels_wise_image_patches_matrix(image, kernel_shape)

    assert np.all(expected == actual)