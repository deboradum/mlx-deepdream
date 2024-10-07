from collections import namedtuple
from mlxim.model import create_model
from mlx.nn.layers.base import Module

import mlx.nn as nn
import mlx.models as mlx_models


class Resnet50(Module):
    def __init__(self, load_weights=True):
        super().__init__()

        resnet = create_model("resnet50")
        # model = create_model("resnet18", weights="path/to/resnet18/model.safetensors")

        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer10 = resnet.layer1.children()["layers"][0]
        self.layer11 = resnet.layer1.children()["layers"][1]
        self.layer12 = resnet.layer1.children()["layers"][2]
        self.layer20 = resnet.layer2.children()["layers"][0]
        self.layer21 = resnet.layer2.children()["layers"][1]
        self.layer22 = resnet.layer2.children()["layers"][2]
        self.layer23 = resnet.layer2.children()["layers"][3]
        self.layer30 = resnet.layer3.children()["layers"][0]
        self.layer31 = resnet.layer3.children()["layers"][1]
        self.layer32 = resnet.layer3.children()["layers"][2]
        self.layer33 = resnet.layer3.children()["layers"][3]
        self.layer34 = resnet.layer3.children()["layers"][4]
        self.layer35 = resnet.layer3.children()["layers"][5]
        self.layer40 = resnet.layer4.children()["layers"][0]
        self.layer41 = resnet.layer4.children()["layers"][1]
        self.layer42 = resnet.layer4.children()["layers"][2]
        self.layer42_conv1 = resnet.layer4.children()["layers"][2].conv1
        self.layer42_bn1 = resnet.layer4.children()["layers"][2].bn1
        self.layer42_conv2 = resnet.layer4.children()["layers"][2].conv2
        self.layer42_bn2 = resnet.layer4.children()["layers"][2].bn2
        self.layer42_conv3 = resnet.layer4.children()["layers"][2].conv3
        self.layer42_bn3 = resnet.layer4.children()["layers"][2].bn3
        self.layer42_relu = resnet.layer4.children()["layers"][2].relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        layer10 = x
        x = self.layer11(x)
        layer11 = x
        x = self.layer12(x)
        layer12 = x
        x = self.layer20(x)
        layer20 = x
        x = self.layer21(x)
        layer21 = x
        x = self.layer22(x)
        layer22 = x
        x = self.layer23(x)
        layer23 = x
        x = self.layer30(x)
        layer30 = x
        x = self.layer31(x)
        layer31 = x
        x = self.layer32(x)
        layer32 = x
        x = self.layer33(x)
        layer33 = x
        x = self.layer34(x)
        layer34 = x
        x = self.layer35(x)
        layer35 = x
        x = self.layer40(x)
        layer40 = x
        x = self.layer41(x)
        layer41 = x

        layer42_identity = layer41
        x = self.layer42_conv1(x)
        layer420 = x
        x = self.layer42_bn1(x)
        layer421 = x
        x = self.layer42_relu(x)
        layer422 = x
        x = self.layer42_conv2(x)
        layer423 = x
        x = self.layer42_bn2(x)
        layer424 = x
        x = self.layer42_relu(x)
        layer425 = x
        x = self.layer42_conv3(x)
        layer426 = x
        x = self.layer42_bn3(x)
        layer427 = x
        x += layer42_identity
        layer428 = x
        x = self.relu(x)
        layer429 = x

        # Feel free to experiment with different layers.
        net_outputs = namedtuple("ResNet50Outputs", self.layer_names)
        out = net_outputs(layer10, layer23, layer35, layer40)
        # layer35 is my favourite
        return out
