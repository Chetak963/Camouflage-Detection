import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Conv Block
class ConvBlock(nn.Module): # A basic feature extraction unit
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Channel Attention (SE Block) Squeeze-and-Excitation
class SEBlock(nn.Module): # A channel attention mechanism
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        # print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape)
        return x * y


# Improved Reverse Attention
class ReverseAttention(nn.Module): # Focuses on missed regions
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, pred):
        print(pred.shape)
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=False)
        reverse = 1 - pred
        x = x * reverse

        x = self.conv1(x)
        x = self.conv2(x)

        return self.out(x)


# FINAL IMPROVED PRANET
class PraNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Encoder
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer2 = backbone.layer1   # 256
        self.layer3 = backbone.layer2   # 512
        self.layer4 = backbone.layer3   # 1024
        self.layer5 = backbone.layer4   # 2048
        
        

        # Decoder
        self.conv5 = ConvBlock(2048, 256)
        self.conv4 = ConvBlock(1024, 256)
        self.conv3 = ConvBlock(512, 256)
        self.conv2 = ConvBlock(256, 256)
        # print(self.conv2)

        # Attention
        self.se5 = SEBlock(256)
        self.se4 = SEBlock(256)
        self.se3 = SEBlock(256)

        # Reverse Attention
        self.ra4 = ReverseAttention(256)
        self.ra3 = ReverseAttention(256)
        self.ra2 = ReverseAttention(256)

        # Fusion conv (instead of simple +)
        self.fuse4 = ConvBlock(512, 256)
        self.fuse3 = ConvBlock(512, 256)
        self.fuse2 = ConvBlock(512, 256)

        # Output
        self.out = nn.Conv2d(256, 1, 1)

        total_params = sum(p.numel() for p in self.parameters())
        # print(f" Total Parameters: {total_params:,}\n")

    def forward(self, x):

        # Encoder
        x1 = self.layer1(x)
        print("x1: [B, C, H, w]", x1.shape) # Batch, channel, height , width
        x2 = self.layer2(x1)
        print("x2:  [B, C, H, w]", x2.shape)
        x3 = self.layer3(x2)
        print("x3:  [B, C, H, w]", x3.shape)
        x4 = self.layer4(x3)
        print("x4:  [B, C, H, w]", x4.shape)

        x5 = self.layer5(x4)
        print("x5:", x5.shape)


        # Decoder
        d5 = self.se5(self.conv5(x5)) # Process deepest features:
                                        # Reduce channels
                                        # Apply attention
        # print(self.conv5(x5).shape)
        print("d5 step1:", d5.shape)

        d4 = F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=False) # Upsample (increase size)
        print("d4 step2:", d4.shape)
        d4 = torch.cat([self.conv4(x4), d4], dim=1) # Concatenate : encoder feature + decoder feature
        print("d4 step3:", d4.shape)
        d4 = self.fuse4(d4) # Fuse features (learned combination)
        print("d4 step4:", d4.shape)
        d4 = self.ra4(d4, d5) # Reverse Attention : refine using previous prediction
        print("d4 step5:", d4.shape)
        d4 = self.se4(d4)
        # print("d4:", d4.shape)
        print("d4 step6:", d4.shape)


        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([self.conv3(x3), d3], dim=1)
        d3 = self.fuse3(d3)
        d3 = self.ra3(d3, d4)
        d3 = self.se3(d3)

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([self.conv2(x2), d2], dim=1)
        d2 = self.fuse2(d2)
        d2 = self.ra2(d2, d3)
        # print("d2:", d2.shape)


        # Deep supervision outputs
        out2 = self.out(d2) # Main prediction
        # Intermediate outputs
        out3 = F.interpolate(self.out(d3), scale_factor=2, mode='bilinear', align_corners=False)
        out4 = F.interpolate(self.out(d4), scale_factor=4, mode='bilinear', align_corners=False)
        out5 = F.interpolate(self.out(d5), scale_factor=8, mode='bilinear', align_corners=False)

        # Final output
        out = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)

        return out, out2, out3, out4, out5


def get_pranet():
    return PraNet()