import cv2

import utils.utils as utils
from utils.constants import UPPER_IMAGE_BOUND, LOWER_IMAGE_BOUND

import numpy as np
import mlx.core as mx


def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    out = model(input_tensor)

    # Step 1: Grab activations/feature maps of interest
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        loss_component = torch.nn.MSELoss(reduction="mean")(
            layer_activation, mx.zeros_like(layer_activation)
        )
        losses.append(loss_component)

    loss = mx.mean(mx.stack(losses))
    loss.backward()

    # Step 3: Process image gradients (smoothing + normalization)
    grad = input_tensor.grad.data

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config["num_gradient_ascent_iterations"]) * 2.0 + config[
        "smoothing_coefficient"
    ]
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(
        grad
    )  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = mx.std(smooth_grad)
    g_mean = mx.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += config["lr"] * smooth_grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = mx.max(
        mx.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND
    )


def deepdream_img(config, img):
    model = utils.get_model(config["model_name"])
    try:
        layer_ids_to_use = [
            model.layer_names.index(layer_name)
            for layer_name in config["layers_to_use"]
        ]
    except Exception as e:
        print(f'Invalid layer names {[name for name in config["layers_to_use"]]}.')
        print(
            f'Available layers for model {config["model_name"]} are {model.layer_names}.'
        )
        return
    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]
    for pyramid_level in range(config["pyramid_size"]):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.mlx_input_adapter(img)

        for iteration in range(config["num_gradient_ascent_iterations"]):
            h_shift, w_shift = np.random.randint(
                -config["spatial_shift_size"], config["spatial_shift_size"] + 1, 2
            )
            input_tensor = utils.random_circular_spatial_shift(
                input_tensor, h_shift, w_shift
            )

            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)

            input_tensor = utils.random_circular_spatial_shift(
                input_tensor, h_shift, w_shift, should_undo=True
            )

        img = utils.mlx_output_adapter(input_tensor)

    return utils.post_process_numpy_img(img)


if __name__ == "__main__":
    config = {
        "model_name": "vgg19",
        "layers_to_use": ["relu4_3"],
        "pyramid_size": 4,
        "pyramid_ratio": 1.3,
        "num_gradient_ascent_iterations": 4,
        "spatial_shift_size": 32,
        "img_path": "figures.jpg",
        "img_width": 1920,
    }
    img = utils.load_image(config["img_path"], target_shape=config["img_width"])
    img = deepdream_img(config, img=img)
