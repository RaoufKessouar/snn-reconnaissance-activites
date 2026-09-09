import torch
import torch.nn as nn
# ANN-BN : noms de classes conserves pour compat avec model.py.
# ConvBNTTLIFBlock == Conv -> BN classique -> ReLU (aucun etat temporel).

class ConvBNTTLIFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps, stride=1, padding=1, kernel_size=3, tau=8.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1, affine=True)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, t):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps, kernel_size=3, stride=1, tau=8.0):
        super().__init__()
        self.conv1 = ConvBNTTLIFBlock(in_channels, out_channels, num_steps, stride=stride, padding=kernel_size//2, kernel_size=kernel_size)
        self.conv2 = ConvBNTTLIFBlock(out_channels, out_channels, num_steps, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.needs_downsampling = stride != 1 or in_channels != out_channels
        if self.needs_downsampling:
            self.downsampling_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.downsampling_bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1, affine=True)
    def forward(self, x, t):
        identity = x
        out = self.conv1(x, t)
        out = self.conv2(out, t)
        if self.needs_downsampling:
            identity = self.downsampling_bn(self.downsampling_conv(identity))
        return out + identity
