import cv2
import shutil
import time
import os

from parser import parse_args
from utils.constants import (
    UPPER_IMAGE_BOUND,
    LOWER_IMAGE_BOUND,
    SUPPORTED_VIDEO_FORMATS,
    BINARIES_PATH,
    OUT_IMAGES_PATH,
    OUT_VIDEOS_PATH,
    OUT_GIF_PATH,
)

import numpy as np
import mlx.nn as nn
import mlx.core as mx
import utils.utils as utils
import utils.video_utils as video_utils


# Make sure these exist as the rest of the code relies on it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)
os.makedirs(OUT_VIDEOS_PATH, exist_ok=True)
os.makedirs(OUT_GIF_PATH, exist_ok=True)


def gradient_ascent(config, model, input_array, layer_ids_to_use, iteration):
    start1 = time.perf_counter()

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
    stop1 = time.perf_counter()
    print("\tGetting grads took", round(stop1 - start1, 4), "seconds")

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config["num_gradient_ascent_iterations"]) * 2.0 + config[
        "smoothing_coefficient"
    ]
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma).forward(
        grads
    )  # "magic number" 9 just works well

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
    start1 = time.perf_counter()
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
        startp = time.perf_counter()
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
        stopp = time.perf_counter()
        print("pyramid level processing took", round(stopp - startp, 2), "seconds")
    stop1 = time.perf_counter()
    print("result took", round(stop1 - start1, 4), "seconds")
    return utils.post_process_numpy_img(img)


def deep_dream_video_ouroboros():
    raise NotImplementedError


def deep_dream_video(config):
    video_path = utils.parse_input_file(config["input"])
    tmp_input_dir = os.path.join(OUT_VIDEOS_PATH, "tmp_input")
    tmp_output_dir = os.path.join(OUT_VIDEOS_PATH, "tmp_out")
    config["dump_dir"] = tmp_output_dir
    os.makedirs(tmp_input_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)

    metadata = video_utils.extract_frames(video_path, tmp_input_dir)
    config["fps"] = metadata["fps"]
    utils.print_deep_dream_video_header(config)

    last_img = None
    for frame_id, frame_name in enumerate(sorted(os.listdir(tmp_input_dir))):
        # Step 1: load the video frame
        print(f"Processing frame {frame_id}")
        frame_path = os.path.join(tmp_input_dir, frame_name)
        frame = utils.load_image(frame_path, target_shape=config["img_width"])

        # Step 2: potentially blend it with the last frame
        if config["blend"] is not None and last_img is not None:
            # blend: 1.0 - use the current frame, 0.0 - use the last frame, everything in between will blend the two
            frame = utils.linear_blend(last_img, frame, config["blend"])

        # Step 3: Send the blended frame to some good old DeepDreaming
        dreamed_frame = deepdream_image(config, frame)

        # Step 4: save the frame and keep the reference
        last_img = dreamed_frame
        dump_path = utils.save_and_maybe_display_image(
            config, dreamed_frame, name_modifier=frame_id
        )
        print(f"Saved DeepDream frame to: {os.path.relpath(dump_path)}\n")

    video_utils.create_video_from_intermediate_results(config)

    shutil.rmtree(tmp_input_dir)  # remove tmp files
    print(f"Deleted tmp frame dump directory {tmp_input_dir}.")


def get_config(args):
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config["dump_dir"] = (
        OUT_VIDEOS_PATH if config["create_ouroboros"] else OUT_IMAGES_PATH
    )
    config["dump_dir"] = os.path.join(
        # config["dump_dir"], f'{config["model_name"]}_{config["pretrained_weights"]}'
        config["dump_dir"],
        f'{config["model_name"]}sx',
    )
    config["input_name"] = os.path.basename(config["input"])

    return config


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args)

    # Create Ouroboros video (feeding neural network's output to it's input)
    if config["create_ouroboros"]:
        deep_dream_video_ouroboros(config)
    # Create a blended DeepDream video
    elif any(
        [
            config["input_name"].lower().endswith(video_ext)
            for video_ext in SUPPORTED_VIDEO_FORMATS
        ]
    ):  # only support mp4 atm
        deep_dream_video(config)
    else:  # Create a static DeepDream image
        print("Dreaming started!")
        img = deepdream_image(
            config, img=None
        )  # img=None -> will be loaded inside of deep_dream_static_image
        dump_path = utils.save_and_maybe_display_image(config, img)
        print(f"Saved DeepDream static image to: {os.path.relpath(dump_path)}\n")
