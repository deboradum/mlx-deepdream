import numpy as np
import mlx.core as mx


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOWER_IMAGE_BOUND = mx.array((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1))
UPPER_IMAGE_BOUND = mx.array(
    ((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)
)
