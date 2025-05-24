import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.components import Conv, C2f


class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        self.topdown1 = C2f(int(768 * w), int(256 * w), n, shortcut=False)
        self.topdown2 = C2f(int(512 * w * (1 + r)), int(512 * w), n, shortcut=False)

        self.downsample0 = Conv(int(256 * w), int(256 * w), self.kernelSize, self.stride)
        self.downsample1 = Conv(int(512 * w), int(512 * w), self.kernelSize, self.stride)

        self.bottomup0 = C2f(int(768 * w), int(512 * w), n, shortcut=False)
        self.bottomup1 = C2f(int(512 * w * (1 + r)), int(512 * w * r), n, shortcut=False)

        # raise NotImplementedError("Neck::__init__")

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                print(f"[!] NaN found in neck.{name}")
        up_feat3 = F.interpolate(feat3, size=feat2.shape[2:], mode="nearest")
        check_nan(up_feat3, "up_feat3")
        up_feat3_concat_feat2 = torch.cat((up_feat3, feat2), dim=1)
        check_nan(up_feat3_concat_feat2, "up_feat3_concat_feat2")
        print(f"up_feat3_concat_feat2 shape: {up_feat3_concat_feat2.shape}")
        up_feat3_concat_feat2 *= 0.01
        C = self.topdown2(up_feat3_concat_feat2)
        check_nan(C, "C")
        up_C = F.interpolate(C, size=feat1.shape[2:], mode="nearest")
        check_nan(up_C, "up_C")
        up_C_concat_feat1 = torch.cat((up_C, feat1), dim=1)
        check_nan(up_C_concat_feat1, "up_C_concat_feat1")
        X = self.topdown1(up_C_concat_feat1)
        check_nan(X, "X")
        down_X = self.downsample0(X)
        check_nan(down_X, "down_X")
        down_X_concat_C = torch.cat((down_X, C), dim=1)
        check_nan(down_X_concat_C, "down_X_concat_C")
        Y = self.bottomup0(down_X_concat_C)
        check_nan(Y, "Y")
        down_Y = self.downsample1(Y)
        check_nan(down_Y, "down_Y")
        if down_Y.shape[2:] != feat3.shape[2:]:
            down_Y = F.interpolate(down_Y, size=feat3.shape[2:], mode="nearest")
        down_Y_concat_feat3 = torch.cat((down_Y, feat3), dim=1)
        check_nan(down_Y_concat_feat3, "down_Y_concat_feat3")
        Z = self.bottomup1(down_Y_concat_feat3)
        check_nan(Z, "Z")

        return C, X, Y, Z

        #raise NotImplementedError("Neck::forward")
