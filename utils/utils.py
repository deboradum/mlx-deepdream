import os
import cv2
import numbers

import numpy as np
import mlx.nn as nn
import mlx.core as mx

from models.Vgg19 import Vgg19
from mlx.nn.layers.base import Module
from utils.constants import IMAGENET_MEAN_1, IMAGENET_STD_1


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f"Path does not exist: {img_path}")
    img = cv2.imread(img_path)[
        :, :, ::-1
    ]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if (
            isinstance(target_shape, int) and target_shape != -1
        ):  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
        else:  # set both dimensions to target shape
            img = cv2.resize(
                img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC
            )

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def get_model(model_type):
    if model_type == "vgg19":
        model = Vgg19()
    # elif model_type == "alexnet":
    #     model = mlx_models.vision.AlexNet(load_weights=True)
    else:
        raise Exception(f"Model '{model_type}' not supported")

    return model


def pre_process_numpy_img(img):
    assert isinstance(img, np.ndarray), f"Expected numpy image got {type(img)}"

    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1  # normalize image
    return img


def get_new_shape(config, base_shape, pyramid_level):
    SHAPE_MARGIN = 10
    pyramid_ratio = config["pyramid_ratio"]
    pyramid_size = config["pyramid_size"]
    exponent = pyramid_level - pyramid_size + 1
    new_shape = np.round(np.float32(base_shape) * (pyramid_ratio**exponent)).astype(
        np.int32
    )

    if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
        print(
            f'Pyramid size {config["pyramid_size"]} with pyramid ratio {config["pyramid_ratio"]} gives too small pyramid levels with size={new_shape}'
        )
        exit(0)

    return new_shape


def mlx_input_adapter(img):
    # shape = (1, 3, H, W)
    # CHANGED TO
    # shape = (1, H, W, 3)
    array = mx.expand_dims(mx.array(img), 0)
    # array.requires_grad = True # TODO
    return array


def mlx_output_adapter(array):
    # Push to CPU, detach from the computational graph, convert from (1, H, W, 3) into (H, W, 3)
    array = np.array(array[0])
    return array


def random_circular_spatial_shift(array, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    # with torch.no_grad():
    rolled = mx.array(
        np.roll(
            np.array(array), shift=(h_shift, w_shift), axis=(1, 2)
        )  # there's no MLX roll?
    )
    # rolled.requires_grad = True
    return rolled


class CascadeGaussianSmoothing(Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """

    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers
        sigmas = [
            [coeff * sigma, coeff * sigma] for coeff in cascade_coefficients
        ]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = mx.meshgrid(
            [mx.arange(size, dtype=mx.float32) for size in kernel_size]
        )
        for sigma in sigmas:
            kernel = mx.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= (
                    1
                    / (std_1d * mx.sqrt(2 * mx.pi))
                    * mx.exp(-(((grid - mean) / std_1d) ** 2) / 2)
                )
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / mx.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = nn.Conv2d

    def forward(self, input):
        input = mx.pad(
            input, [self.pad, self.pad, self.pad, self.pad], mode="constant"
        )  # mode=reflect

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3