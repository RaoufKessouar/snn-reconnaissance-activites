import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron


import os

class _ReLUNeuron(nn.Module):
    def forward(self, x): return torch.relu(x)
    def reset(self): pass

def make_lif_node(tau=8.0, v_threshold=1.0):
    kind = os.environ.get("NEURON", "lif").lower()
    if kind == "relu":
        return _ReLUNeuron()
    if kind == "qif":
        return neuron.QIFNode(tau=tau, v_c=0.8, a0=1.0, v_threshold=v_threshold,
                              v_rest=0.0, v_reset=-0.1)
    if kind in ("plif", "alif"):
        try:
            return neuron.ParametricLIFNode(init_tau=tau, decay_input=False,
                                            v_threshold=v_threshold, v_reset=None)
        except TypeError:
            return neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=None)
    try:
        return neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                              v_reset=None, decay_input=False)
    except TypeError:
        return neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=None)


class tdBatchNorm2d(nn.Module):
    """Threshold-dependent BN (Zheng et al., AAAI 2021). Entree/sortie: [T,B,C,H,W]."""
    def __init__(self, num_features, alpha=1.0, Vth=1.0, eps=1e-4, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.alpha, self.Vth, self.eps = alpha, Vth, eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def reset_running_stats(self):
        self.running_mean.zero_(); self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()

    def forward(self, x):                      # x: [T,B,C,H,W]
        if self.training:
            mean = x.mean(dim=[0, 1, 3, 4])
            var  = x.var(dim=[0, 1, 3, 4], unbiased=False)
            if self.momentum is None:
                self.num_batches_tracked += 1
                mom = 1.0 / float(self.num_batches_tracked)
            else:
                mom = self.momentum
            with torch.no_grad():
                self.running_mean.mul_(1 - mom).add_(mom * mean)
                self.running_var.mul_(1 - mom).add_(mom * var)
            um, uv = mean, var
        else:
            um, uv = self.running_mean, self.running_var
        xhat = self.alpha * self.Vth * (x - um[None, None, :, None, None]) \
               / torch.sqrt(uv[None, None, :, None, None] + self.eps)
        return self.weight[None, None, :, None, None] * xhat \
             + self.bias[None, None, :, None, None]


class ConvTdBNLIFBlock(nn.Module):
    """Conv (temps plie dans le batch) -> tdBN (sur T,B) -> LIF (deroule sur T)."""
    def __init__(self, in_channels, out_channels, num_steps, stride=1,
                 padding=1, kernel_size=3, tau=8.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.tdbn = tdBatchNorm2d(out_channels, Vth=1.0)
        self.neuron = make_lif_node(tau=tau)

    def forward(self, x):                      # x: [T,B,Cin,H,W]
        T, B = x.shape[0], x.shape[1]
        y = self.conv(x.flatten(0, 1))         # [T*B,Cout,H',W']
        y = y.view(T, B, *y.shape[1:])         # [T,B,Cout,H',W']
        y = self.tdbn(y)
        self.neuron.reset()
        out = [self.neuron(y[t]) for t in range(T)]
        return torch.stack(out, dim=0)         # [T,B,Cout,H',W']


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps,
                 kernel_size=3, stride=1, tau=8.0):
        super().__init__()
        self.conv1 = ConvTdBNLIFBlock(in_channels, out_channels, num_steps,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=kernel_size // 2, tau=tau)
        self.conv2 = ConvTdBNLIFBlock(out_channels, out_channels, num_steps,
                                      kernel_size=kernel_size, stride=1,
                                      padding=kernel_size // 2, tau=tau)
        self.needs_downsampling = stride != 1 or in_channels != out_channels
        if self.needs_downsampling:
            self.downsampling_conv = nn.Conv2d(in_channels, out_channels,
                                               kernel_size=1, stride=stride, bias=False)
            self.downsampling_tdbn = tdBatchNorm2d(out_channels, Vth=1.0)

    def forward(self, x):                      # [T,B,Cin,H,W]
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.needs_downsampling:
            T, B = identity.shape[0], identity.shape[1]
            idy = self.downsampling_conv(identity.flatten(0, 1))
            idy = idy.view(T, B, *idy.shape[1:])
            identity = self.downsampling_tdbn(idy)
        return out + identity
