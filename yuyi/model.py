import torch
import torch.nn as nn
import torchvision.models as models


class SegmentationModel(nn.Module):
    """Wrapper around a torchvision segmentation model (DeepLabV3) or a
    simple UNet implementation. Defaults to DeepLabV3 with ResNet-50.
    """

    def __init__(self, num_classes: int, backbone: str = "deeplabv3"):
        super().__init__()
        if backbone == "deeplabv3":
            # use pretrained backbone but reinitialize classifier
            self.model = models.segmentation.deeplabv3_resnet50(
                pretrained=False, num_classes=num_classes
            )
        elif backbone == "unet":
            self.model = UNet(num_classes=num_classes)
        else:
            raise ValueError(f"unsupported backbone {backbone}")

    def forward(self, x):
        return self.model(x)


# a minimal UNet implementation
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # down part
        in_ch = input_channels
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        # up part
        for f in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(in_ch, f))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(2)(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
