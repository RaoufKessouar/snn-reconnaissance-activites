import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron


def make_lif_node(tau=8.0, v_threshold=1.0):
    try:
        return neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=None,
            decay_input=False,
        )
    except TypeError:
        return neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=None,
        )


class BNTT(nn.Module):
    def __init__(self, num_features, num_steps):
        super().__init__()
        self.bntt = nn.ModuleList([
            nn.BatchNorm2d(
                num_features=num_features,
                eps=1e-4,
                momentum=0.1,
                affine=True,
            )
            for _ in range(num_steps)
        ])

    def forward(self, x, t):
        return self.bntt[t](x)


class ConvBNTTLIFBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_steps,
        stride=1,
        padding=1,
        kernel_size=3,
        tau=8.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bntt = BNTT(out_channels, num_steps)
        self.neuron = make_lif_node(tau=tau)

    def forward(self, x, t):
        out = self.conv(x)
        out = self.bntt(out, t)
        out = self.neuron(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_steps,
        kernel_size=3,
        stride=1,
        tau=8.0,
    ):
        super().__init__()

        self.conv1 = ConvBNTTLIFBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_steps=num_steps,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            tau=tau,
        )

        self.conv2 = ConvBNTTLIFBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            num_steps=num_steps,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            tau=tau,
        )

        self.needs_downsampling = stride != 1 or in_channels != out_channels

        if self.needs_downsampling:
            self.downsampling_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.downsampling_bntt = BNTT(out_channels, num_steps)

    def forward(self, x, t):
        identity = x

        out = self.conv1(x, t)
        out = self.conv2(out, t)

        if self.needs_downsampling:
            identity = self.downsampling_conv(identity)
            identity = self.downsampling_bntt(identity, t)

        out = out + identity
        return out