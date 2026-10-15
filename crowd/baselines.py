"""Independent MCNN/CSRNet architecture reproductions for our matched split.

MCNN: Zhang et al., CVPR 2016, Fig. 3.
CSRNet: Li et al., CVPR 2018; architecture checked against the author's
https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/model.py.
These are new implementations, not copies of published trained checkpoints.
The standardized data/augmentation/epoch recipe is explicitly archived.
"""
import torch
from torch import nn


def conv_layers(channels, kernel=3, dilation=1, incoming=3):
    layers = []
    for outgoing in channels:
        if outgoing == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers.extend((nn.Conv2d(incoming, outgoing, kernel, padding=(kernel // 2) * dilation, dilation=dilation), nn.ReLU(inplace=True)))
            incoming = outgoing
    return nn.Sequential(*layers)


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.columns = nn.ModuleList()
        for widths, kernels in (((16, 32, 16, 8), (9, 7, 7, 7)),
                               ((20, 40, 20, 10), (7, 5, 5, 5)),
                               ((24, 48, 24, 12), (5, 3, 3, 3))):
            layers, incoming = [], 3
            for index, (width, kernel) in enumerate(zip(widths, kernels)):
                layers.extend((nn.Conv2d(incoming, width, kernel, padding=kernel // 2), nn.ReLU(inplace=True)))
                if index < 2:
                    layers.append(nn.MaxPool2d(2))
                incoming = width
            self.columns.append(nn.Sequential(*layers))
        self.output_layer = nn.Conv2d(30, 1, 1)
        initialize(self)

    def forward(self, images):
        return self.output_layer(torch.cat([column(images) for column in self.columns], dim=1))


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = conv_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512])
        self.backend = conv_layers([512, 512, 512, 256, 128, 64], incoming=512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, 1)
        initialize(self)

    def forward(self, images):
        return self.output_layer(self.backend(self.frontend(images)))

    def load_vgg_frontend(self, path=None):
        if path:
            from .protocol import sha256_file
            if not sha256_file(path).startswith("397923af"):
                raise ValueError("CSRNet 前端需要官方 VGG16 ImageNet V1 权重（SHA-256 前缀 397923af）")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint.get("state_dict", checkpoint)
        else:
            from torchvision.models import VGG16_Weights
            state = VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        # Only take the pretrained frontend: no crowd test-trained model is loaded.
        self.frontend.load_state_dict({key: state[f"features.{key}"] for key in self.frontend.state_dict()}, strict=True)


def initialize(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)


def baseline_input(images, name):
    images = images.float() / 255.
    if name == "csrnet":
        mean = images.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = images.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def make_baseline(name):
    return {"mcnn": MCNN, "csrnet": CSRNet}[name]()
