from collections import namedtuple
from mlx.nn.layers.base import Module

import mlx.nn as nn
import mlx.core as mx
import mlx.models as mlx_models



class Vgg19(Module):
    def __init__(self, load_weights=True):
        super().__init__()

        vgg19 = mlx_models.vision.VGG19(load_weights=load_weights)
        vgg_pretrained_features = vgg19.features.children()["layers"]

        self.layer_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

        self.slice1 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        # self.slice1.
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)])

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x

        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out
