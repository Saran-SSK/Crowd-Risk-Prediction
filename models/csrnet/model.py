import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        self.frontend_feat = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512
        ]

        self.backend_feat = [
            512, 512, 512,
            256, 128, 64
        ]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat,
            in_channels=512,
            dilation=True
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        # Load pretrained VGG16 weights only when we're NOT loading CSRNet weights
        if not load_weights:
            self._load_vgg16_weights()

    def _load_vgg16_weights(self):
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        frontend_state = self.frontend.state_dict()
        vgg_state = vgg.features.state_dict()

        for key in frontend_state.keys():
            frontend_state[key].copy_(vgg_state[key])

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):

    layers = []

    d_rate = 2 if dilation else 1

    for layer in cfg:

        if layer == 'M':
            layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        else:

            conv = nn.Conv2d(
                in_channels,
                layer,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate
            )

            if batch_norm:
                layers.extend([
                    conv,
                    nn.BatchNorm2d(layer),
                    nn.ReLU(inplace=True)
                ])
            else:
                layers.extend([
                    conv,
                    nn.ReLU(inplace=True)
                ])

            in_channels = layer

    return nn.Sequential(*layers)