import torch.nn as nn
from model.base.components import Conv, C2f, SPPF


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        self.stem = Conv(self.imageChannel, int(64 * w), self.kernelSize, self.stride)
        self.stage1 = nn.Sequential(
            Conv(int(64 * w), int(128 * w), self.kernelSize, self.stride),
            C2f(int(128 * w), int(128 * w), n, shortcut=True),
        )
        self.stage2 = nn.Sequential(
            Conv(int(128 * w), int(256 * w), self.kernelSize, self.stride),
            C2f(int(256 * w), int(256 * w), n * 2, shortcut=True),
        )
        self.stage3 = nn.Sequential(
            Conv(int(256 * w), int(512 * w), self.kernelSize, self.stride),
            C2f(int(512 * w), int(512 * w), n * 2, shortcut=True),
        )
        self.stage4 = nn.Sequential(
            Conv(int(512 * w), int(512 * w * r), self.kernelSize, self.stride),
            C2f(int(512 * w * r), int(512 * w * r), n, shortcut=True),
            SPPF(int(512 * w * r), int(512 * w * r), 5),
        )

        # raise NotImplementedError("Backbone::__init__")

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """
        x = self.stem(x)
        feat0 = self.stage1(x)
        feat1 = self.stage2(feat0)
        feat2 = self.stage3(feat1)
        feat3 = self.stage4(feat2)
        return feat0, feat1, feat2, feat3

        # raise NotImplementedError("Backbone::forward")
