import torch
import torch.nn as nn
from .block_tdbn import ConvTdBNLIFBlock, ResidualBlock


class SResNest(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, num_steps=60,
                 num_classes=81, n=6, leak_mem=0.874, kernel_size=3):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.n = n
        tau = 1.0 / (1.0 - leak_mem)
        self.conv_init = ConvTdBNLIFBlock(in_channels, base_channels, num_steps,
                                          stride=1, kernel_size=kernel_size,
                                          padding=kernel_size // 2, tau=tau)
        self.stage1 = self._make_stage(base_channels, base_channels, n, 1,
                                       num_steps, kernel_size, tau)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, n, 2,
                                       num_steps, kernel_size, tau)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, n, 2,
                                       num_steps, kernel_size, tau)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes, bias=False)
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, first_stride,
                    num_steps, kernel_size, tau):
        blocks = [ResidualBlock(in_channels, out_channels, num_steps,
                                kernel_size, first_stride, tau)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, num_steps,
                                        kernel_size, 1, tau))
        return nn.ModuleList(blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=2.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0)

    def _forward_stage(self, x, stage):
        out = x
        for block in stage:
            out = block(out)
        return out

    def forward(self, x):                      # x: [B,T,2,H,W]
        T = x.shape[1]
        if T > self.num_steps:
            raise ValueError(f"Input T={T}, but model num_steps={self.num_steps}")
        x = x.permute(1, 0, 2, 3, 4).contiguous()   # [T,B,2,H,W]
        out = self.conv_init(x)
        out = self._forward_stage(out, self.stage1)
        out = self._forward_stage(out, self.stage2)
        out = self._forward_stage(out, self.stage3)  # [T,B,C,H,W]
        T2, B = out.shape[0], out.shape[1]
        out = self.pool(out.flatten(0, 1))           # [T*B,C,1,1]
        out = torch.flatten(out, 1)                  # [T*B,C]
        out = self.fc(out)                           # [T*B,num_classes]
        return out.view(T2, B, -1).mean(dim=0)       # moyenne sur T -> [B,classes]
