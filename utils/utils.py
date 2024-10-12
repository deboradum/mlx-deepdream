import os
import cv2
import numbers

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

from models.Vgg19 import Vgg19
from models.Alexnet import Alexnet
from models.Resnet import Resnet50
from mlx.nn.layers.base import Module
from utils.constants import (
    IMAGENET_MEAN_1,
    IMAGENET_STD_1,
    SupportedModels,
    INPUT_DATA_PATH,
)


def parse_input_file(input):
    # Handle abs/rel paths
    if os.path.exists(input):
        return input
    # If passed only a name and it doesn't exist in the current working dir assume it's in input data dir
    elif os.path.exists(os.path.join(INPUT_DATA_PATH, input)):
        return os.path.join(INPUT_DATA_PATH, input)
    else:
        raise Exception(f"Input path {input} is not valid.")


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
    if model_type == SupportedModels.VGG19.name:
        model = Vgg19()
    elif model_type == SupportedModels.ALEXNET.name:
        model = Alexnet()
    elif model_type == SupportedModels.RESNET50.name:
        model = Resnet50()
    else:
        raise Exception(f"Model '{model_type}' not supported")

    return model


def pre_process_numpy_img(img):
    assert isinstance(img, np.ndarray), f"Expected numpy image got {type(img)}"

    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1  # normalize image
    return img


def post_process_numpy_img(img):
    assert isinstance(img, np.ndarray), f"Expected numpy image got {type(img)}"
    if img.shape[0] == 3:  # if channel-first format move to channel-last (CHW -> HWC)
        img = np.moveaxis(img, 0, 2)

    mean = IMAGENET_MEAN_1.reshape(1, 1, -1)
    std = IMAGENET_STD_1.reshape(1, 1, -1)
    img = (img * std) + mean  # de-normalize
    img = np.clip(img, 0.0, 1.0)  # make sure it's in the [0, 1] range

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
    array = mx.expand_dims(mx.array(img), 0)
    return array


def mlx_output_adapter(array):
    # Push to CPU, detach from the computational graph, convert from (1, H, W, 3) into (H, W, 3)
    array = np.array(array[0])
    return array


def random_circular_spatial_shift(array, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift

    rolled = mx.roll(
        array, (h_shift, w_shift), (1, 2)
    )

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
        meshgrids = mx.meshgrid(*[mx.arange(size) for size in kernel_size])

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
            kernel = mx.reshape(kernel, (1, *kernel.shape, 1))
            kernel = mx.repeat(kernel, 3, 0)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = mx.conv2d

    def forward(self, input):
        input = mx.pad(
            input,
            [(0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)],
            mode="edge",
        )  # mode=reflect

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[3]
        grad1 = self.conv(input, self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3


def build_image_name(config):
    input_name = (
        "rand_noise" if config["use_noise"] else config["input_name"].rsplit(".", 1)[0]
    )
    layers = "_".join(config["layers_to_use"])
    # Looks awful but makes the creation process transparent for other creators
    img_name = f'{input_name}_width_{config["img_width"]}_model_{config["model_name"]}_{layers}_pyrsize_{config["pyramid_size"]}_pyrratio_{config["pyramid_ratio"]}_iter_{config["num_gradient_ascent_iterations"]}_lr_{config["lr"]}_shift_{config["spatial_shift_size"]}_smooth_{config["smoothing_coefficient"]}.jpg'
    return img_name


def save_and_maybe_display_image(config, dump_img, name_modifier=None):
    assert isinstance(
        dump_img, np.ndarray
    ), f"Expected numpy array got {type(dump_img)}."

    # step1: figure out the dump dir location
    dump_dir = config["dump_dir"]
    os.makedirs(dump_dir, exist_ok=True)

    # step2: define the output image name
    if name_modifier is not None:
        dump_img_name = str(name_modifier).zfill(6) + ".jpg"
    else:
        dump_img_name = build_image_name(config)

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img * 255).astype(np.uint8)

    # step3: write image to the file system
    # ::-1 because opencv expects BGR (and not RGB) format...
    dump_path = os.path.join(dump_dir, dump_img_name)
    cv2.imwrite(dump_path, dump_img[:, :, ::-1])

    # step4: potentially display/plot the image
    if config["should_display"]:
        plt.imshow(dump_img)
        plt.show()

    return dump_path


def linear_blend(img1, img2, alpha=0.5):
    return img1 + alpha * (img2 - img1)


def print_deep_dream_video_header(config):
    print(
        f'Creating a DeepDream video from {config["input_name"]}, via {config["model_name"]} model.'
    )
    # print(f'Using pretrained weights = {config["pretrained_weights"]}')
    print(f'Using model layers = {config["layers_to_use"]}')
    print(f'Using lending coefficient = {config["blend"]}.')
    print(f'Video output width = {config["img_width"]}')
    print(f'fps = {config["fps"]}')
    print("*" * 50, "\n")


def print_ouroboros_video_header(config):
    print(
        f'Creating a {config["ouroboros_length"]}-frame Ouroboros video from {config["input_name"]}, via {config["model_name"]} model.'
    )
    print(f'Using {config["frame_transform"]} for the frame transform')
    # print(f'Using pretrained weights = {config["pretrained_weights"]}')
    print(f'Using model layers = {config["layers_to_use"]}')
    print(f'Video output width = {config["img_width"]}')
    print(f'fps = {config["fps"]}')
    print("*" * 50, "\n")