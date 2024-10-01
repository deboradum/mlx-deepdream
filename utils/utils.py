import numpy as np
import mlx.core as mx
import mlx.models as mlx_models

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def mlx_input_adapter(img, device):
    # shape = (1, 3, H, W)
    # tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    # tensor.requires_grad = True
    # return tensor
    print("TODO")


def mlx_output_adapter(tensor):
    # Push to CPU, detach from the computational graph, convert from (1, 3, H, W) into (H, W, 3)
    return np.moveaxis(np.array(tensor), 0, 2)


def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    # with torch.no_grad():
    rolled = mx.array(
        np.array(tensor).roll(tensor, shift=(h_shift, w_shift), axis=(2, 3)) # there's no MLX roll?
    )
    # rolled.requires_grad = True
    return rolled
