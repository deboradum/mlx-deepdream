import argparse

from utils.constants import (
    SupportedModels,
    TRANSFORMS,
)


def parse_args():
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
    parser.add_argument(
        "--frame_transform",
        choices=[t.name for t in TRANSFORMS],
        help="Transform used to transform the output frame and feed it back to the network input",
        default=TRANSFORMS.ZOOM_ROTATE.name,
    )

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

    return args
