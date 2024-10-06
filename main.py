import argparse
import cv2
import os

import utils.utils as utils
from utils.constants import (
    UPPER_IMAGE_BOUND,
    LOWER_IMAGE_BOUND,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    BINARIES_PATH,
    DATA_DIR_PATH,
    INPUT_DATA_PATH,
    OUT_IMAGES_PATH,
    OUT_VIDEOS_PATH,
    OUT_GIF_PATH,
    SupportedModels,
)

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# Make sure these exist as the rest of the code relies on it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)
os.makedirs(OUT_VIDEOS_PATH, exist_ok=True)
os.makedirs(OUT_GIF_PATH, exist_ok=True)


def gradient_ascent(config, model, input_array, layer_ids_to_use, iteration):

    def deepdream_loss(input_array):
        # Step 0: Feed forward pass
        out = model.forward(input_array)

        # Step 1: Grab activations/feature maps of interest
        activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

        # Step 2: Calculate loss over activations
        losses = []
        for layer_activation in activations:
            # Using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
            # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
            loss_component = nn.losses.mse_loss(
                layer_activation, mx.zeros_like(layer_activation), reduction="mean"
            )
            losses.append(loss_component)

        return mx.mean(mx.stack(losses))

    lvalue, grads = mx.value_and_grad(deepdream_loss)(input_array)

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config["num_gradient_ascent_iterations"]) * 2.0 + config[
        "smoothing_coefficient"
    ]
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma).forward(
        grads
    )  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = mx.std(smooth_grad)
    g_mean = mx.mean(smooth_grad)
    smooth_grad = (smooth_grad - g_mean) / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_array += config["lr"] * smooth_grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    # input_array.grad.data.zero_()
    input_array = mx.maximum(
        mx.minimum(input_array, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND
    )


def deepdream_image(config, img):
    model = utils.get_model(config["model_name"])
    try:
        layer_ids_to_use = [
            model.layer_names.index(layer_name)
            for layer_name in config["layers_to_use"]
        ]
    except Exception as _:
        print(f'Invalid layer names {[name for name in config["layers_to_use"]]}.')
        print(
            f'Available layers for model {config["model_name"]} are {model.layer_names}.'
        )
        return

    if img is None:  # load either the provided image or start from a pure noise image
        img_path = utils.parse_input_file(config["input"])
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = utils.load_image(img_path, target_shape=config["img_width"])
        if config["use_noise"]:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]
    for pyramid_level in range(config["pyramid_size"]):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        input_array = utils.mlx_input_adapter(img)

        for iteration in range(config["num_gradient_ascent_iterations"]):
            h_shift, w_shift = np.random.randint(
                -config["spatial_shift_size"], config["spatial_shift_size"] + 1, 2
            )
            input_array = utils.random_circular_spatial_shift(
                input_array, h_shift, w_shift
            )

            gradient_ascent(config, model, input_array, layer_ids_to_use, iteration)

            input_array = utils.random_circular_spatial_shift(
                input_array, h_shift, w_shift, should_undo=True
            )

        img = utils.mlx_output_adapter(input_array)

    return utils.post_process_numpy_img(img)


if __name__ == "__main__":
    # Only a small subset is exposed by design to avoid cluttering
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument(
        "--input",
        type=str,
        help="Input IMAGE or VIDEO name that will be used for dreaming",
        default="figures.jpg",
    )
    parser.add_argument(
        "--img_width", type=int, help="Resize input image to this width", default=1920
    )
    parser.add_argument(
        "--layers_to_use",
        type=str,
        nargs="+",
        help="Layer whose activations we should maximize while dreaming",
        default=["relu4_3"],
    )
    parser.add_argument(
        "--model_name",
        choices=[m.name for m in SupportedModels],
        help="Neural network (model) to use for dreaming",
        default=SupportedModels.VGG19.name,
    )
    # parser.add_argument(
    #     "--pretrained_weights",
    #     choices=[pw.name for pw in SupportedPretrainedWeights],
    #     help="Pretrained weights to use for the above model",
    #     default=SupportedPretrainedWeights.IMAGENET.name,
    # )

    # Main params for experimentation (especially pyramid_size and pyramid_ratio)
    parser.add_argument(
        "--pyramid_size",
        type=int,
        help="Number of images in an image pyramid",
        default=4,
    )
    parser.add_argument(
        "--pyramid_ratio",
        type=float,
        help="Ratio of image sizes in the pyramid",
        default=1.3,
    )
    parser.add_argument(
        "--num_gradient_ascent_iterations",
        type=int,
        help="Number of gradient ascent iterations",
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate i.e. step size in gradient ascent",
        default=0.09,
    )

    # deep_dream_video_ouroboros specific arguments (ignore for other 2 functions)
    parser.add_argument(
        "--create_ouroboros",
        action="store_true",
        help="Create Ouroboros video (default False)",
    )
    parser.add_argument(
        "--ouroboros_length",
        type=int,
        help="Number of video frames in ouroboros video",
        default=30,
    )
    parser.add_argument(
        "--fps", type=int, help="Number of frames per second", default=30
    )
    # parser.add_argument(
    #     "--frame_transform",
    #     choices=[t.name for t in TRANSFORMS],
    #     help="Transform used to transform the output frame and feed it back to the network input",
    #     default=TRANSFORMS.ZOOM_ROTATE.name,
    # )

    # deep_dream_video specific arguments (ignore for other 2 functions)
    parser.add_argument(
        "--blend", type=float, help="Blend coefficient for video creation", default=0.99
    )

    # You usually won't need to change these as often
    parser.add_argument(
        "--should_display",
        action="store_true",
        help="Display intermediate dreaming results (default False)",
    )
    parser.add_argument(
        "--spatial_shift_size",
        type=int,
        help="Number of pixels to randomly shift image before grad ascent",
        default=32,
    )
    parser.add_argument(
        "--smoothing_coefficient",
        type=float,
        help="Directly controls standard deviation for gradient smoothing",
        default=0.5,
    )
    parser.add_argument(
        "--use_noise",
        action="store_true",
        help="Use noise as a starting point instead of input image (default False)",
    )
    args = parser.parse_args()

    # Wrapping configuration into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config["dump_dir"] = (
        OUT_VIDEOS_PATH if config["create_ouroboros"] else OUT_IMAGES_PATH
    )
    config["dump_dir"] = os.path.join(
        # config["dump_dir"], f'{config["model_name"]}_{config["pretrained_weights"]}'
        config["dump_dir"], f'{config["model_name"]}sx'
    )
    config["input_name"] = os.path.basename(config["input"])

    # # Create Ouroboros video (feeding neural network's output to it's input)
    # if config["create_ouroboros"]:
    #     deep_dream_video_ouroboros(config)
    # # Create a blended DeepDream video
    # elif any(
    #     [
    #         config["input_name"].lower().endswith(video_ext)
    #         for video_ext in SUPPORTED_VIDEO_FORMATS
    #     ]
    # ):  # only support mp4 atm
    #     deep_dream_video(config)
    if False:
        pass
    else:  # Create a static DeepDream image
        print("Dreaming started!")
        img = deepdream_image(
            config, img=None
        )  # img=None -> will be loaded inside of deep_dream_static_image
        dump_path = utils.save_and_maybe_display_image(config, img)
        print(f"Saved DeepDream static image to: {os.path.relpath(dump_path)}\n")
