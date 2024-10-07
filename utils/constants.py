import enum
import os

import numpy as np
import mlx.core as mx


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOWER_IMAGE_BOUND = mx.array((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, 1, 1, -1))
UPPER_IMAGE_BOUND = mx.array(
    ((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, 1, 1, -1)
)


class TRANSFORMS(enum.Enum):
    ZOOM = 0
    ZOOM_ROTATE = 1
    TRANSLATE = 2


class SupportedModels(enum.Enum):
    VGG19 = 0
    ALEXNET = 1
    RESNET50 = 2


SUPPORTED_VIDEO_FORMATS = [".mp4"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models", "binaries")
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data")

INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, "input")
OUT_IMAGES_PATH = os.path.join(DATA_DIR_PATH, "out-images")
OUT_VIDEOS_PATH = os.path.join(DATA_DIR_PATH, "out-videos")
OUT_GIF_PATH = os.path.join(OUT_VIDEOS_PATH, "GIFS")
