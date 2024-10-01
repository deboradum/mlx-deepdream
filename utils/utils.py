import os
import cv2

import numpy as np
import mlx.core as mx
import mlx.models as mlx_models

from constants import *

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
        model = mlx_models.vision.VGG19(load_weights=True)
    elif model_type == "alexnet":
        model = mlx_models.vision.AlexNet(load_weights=True)
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
    tensor = mx.expand_dims(mx.array(img), 0)
    # tensor.requires_grad = True # TODO
    return tensor


def mlx_output_adapter(tensor):
    # Push to CPU, detach from the computational graph, convert from (1, H, W, 3) into (H, W, 3)
    tensor = np.array(tensor[0])
    return tensor


def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    # with torch.no_grad():
    rolled = mx.array(
        np.array(tensor).roll(
            tensor, shift=(h_shift, w_shift), axis=(1, 2)
        )  # there's no MLX roll?
    )
    # rolled.requires_grad = True
    return rolled
