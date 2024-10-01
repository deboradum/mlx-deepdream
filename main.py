import cv2

import utils

import numpy as np


def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    print("TODO")
    return


def deepdream_img(config, img):
    model = utils.get_model(config["model_type"])
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
    print("main")
