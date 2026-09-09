import torch
import torch.nn as nn

from .block import ConvBNTTLIFBlock, ResidualBlock


class SResNest(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=32,
        num_steps=60,
        num_classes=81,
        n=6,
        leak_mem=0.874,
        kernel_size=3,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.num_classes = num_classes
        self.n = n

        tau = 1.0 / (1.0 - leak_mem)

        self.conv_init = ConvBNTTLIFBlock(
            in_channels=in_channels,
            out_channels=base_channels,
            num_steps=num_steps,
            stride=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            tau=tau,
        )

        self.stage1 = self._make_stage(
            in_channels=base_channels,
            out_channels=base_channels,
            num_blocks=n,
            first_stride=1,
            num_steps=num_steps,
            kernel_size=kernel_size,
            tau=tau,
        )

        self.stage2 = self._make_stage(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            num_blocks=n,
            first_stride=2,
            num_steps=num_steps,
            kernel_size=kernel_size,
            tau=tau,
        )

        self.stage3 = self._make_stage(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            num_blocks=n,
            first_stride=2,
            num_steps=num_steps,
            kernel_size=kernel_size,
            tau=tau,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes, bias=False)

        self._init_weights()

    def _make_stage(
        self,
        in_channels,
        out_channels,
        num_blocks,
        first_stride,
        num_steps,
        kernel_size,
        tau,
    ):
        blocks = []

        blocks.append(
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                num_steps=num_steps,
                kernel_size=kernel_size,
                stride=first_stride,
                tau=tau,
            )
        )

        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_steps=num_steps,
                    kernel_size=kernel_size,
                    stride=1,
                    tau=tau,
                )
            )

        return nn.ModuleList(blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=2.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0)

    def _forward_stage(self, x, stage, t):
        out = x
        for block in stage:
            out = block(out, t)
        return out

    def forward_single_step(self, x, t):
        out = self.conv_init(x, t)

        out = self._forward_stage(out, self.stage1, t)
        out = self._forward_stage(out, self.stage2, t)
        out = self._forward_stage(out, self.stage3, t)

        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def forward(self, x):
        # x: [B, T, 2, H, W]
        T = x.shape[1]

        if T > self.num_steps:
            raise ValueError(f"Input T={T}, but model num_steps={self.num_steps}")

        out_sum = None

        for t in range(T):
            out_t = self.forward_single_step(x[:, t], t)

            if out_sum is None:
                out_sum = out_t
            else:
                out_sum = out_sum + out_t

        return out_sum / T
