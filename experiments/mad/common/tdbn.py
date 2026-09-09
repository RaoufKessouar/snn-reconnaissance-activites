import torch
import torch.nn as nn

class tdBatchNorm2d(nn.Module):
    """
    Threshold-dependent BN (Zheng et al., AAAI 2021).
    Normalise conjointement sur (T, B, H, W) par canal.
    Entree : x de forme [T, B, C, H, W] (sequence complete).
    """
    def __init__(self, num_features, alpha=1.0, Vth=1.0, eps=1e-4, momentum=0.1):
        super().__init__()
        self.alpha, self.Vth, self.eps, self.momentum = alpha, Vth, eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

    def forward(self, x):                      # x : [T, B, C, H, W]
        if self.training:
            mean = x.mean(dim=[0, 1, 3, 4])
            var  = x.var(dim=[0, 1, 3, 4], unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1-self.momentum).add_(self.momentum*mean)
                self.running_var.mul_(1-self.momentum).add_(self.momentum*var)
        else:
            mean, var = self.running_mean, self.running_var
        xhat = self.alpha * self.Vth * (x - mean[None,None,:,None,None]) \
               / torch.sqrt(var[None,None,:,None,None] + self.eps)
        return self.weight[None,None,:,None,None]*xhat + self.bias[None,None,:,None,None]
