from collections import namedtuple
from mlx.nn.layers.base import Module

import mlx.nn as nn
import mlx.models as mlx_models


class Alexnet(Module):
    def __init__(self, load_weights=True):
        super().__init__()

        alexnet = mlx_models.vision.AlexNet(load_weights=load_weights)
        alexnet_pretrained_features = alexnet.features.children()["layers"]
        self.layer_names = ["relu1", "relu2", "relu3", "relu4", "relu5"]

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        self.slice1 = nn.Sequential(*[alexnet_pretrained_features[x] for x in range(2)])
        self.slice2 = nn.Sequential(
            *[alexnet_pretrained_features[x] for x in range(2, 5)]
        )
        self.slice3 = nn.Sequential(
            *[alexnet_pretrained_features[x] for x in range(5, 8)]
        )
        self.slice4 = nn.Sequential(
            *[alexnet_pretrained_features[x] for x in range(8, 10)]
        )
        self.slice5 = nn.Sequential(
            *[alexnet_pretrained_features[x] for x in range(10, 12)]
        )

    def forward(self, x):
        x = self.slice1(x)
        relu1 = x
        x = self.slice2(x)
        relu2 = x
        x = self.slice3(x)
        relu3 = x
        x = self.slice4(x)
        relu4 = x
        x = self.slice5(x)
        relu5 = x

        alexnet_outputs = namedtuple("alexnetOutputs", self.layer_names)
        out = alexnet_outputs(relu1, relu2, relu3, relu4, relu5)
        return out
