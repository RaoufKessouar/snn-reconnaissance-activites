import torch

from .model_tdbn import SResNest


class SResNestFPP(SResNest):
    """Sortie PAR FRAME : [B, T, num_classes] (detection + localisation)."""
    def forward(self, x):                       # x: [B,T,2,H,W]
        T = x.shape[1]
        if T > self.num_steps:
            raise ValueError(f"Input T={T}, but model num_steps={self.num_steps}")
        x = x.permute(1, 0, 2, 3, 4).contiguous()    # [T,B,2,H,W]
        out = self.conv_init(x)
        out = self._forward_stage(out, self.stage1)
        out = self._forward_stage(out, self.stage2)
        out = self._forward_stage(out, self.stage3)  # [T,B,C,H,W]
        T2, B = out.shape[0], out.shape[1]
        out = self.pool(out.flatten(0, 1))           # [T*B,C,1,1]
        out = torch.flatten(out, 1)                  # [T*B,C]
        out = self.fc(out)                           # [T*B,num_classes]
        return out.view(T2, B, -1).permute(1, 0, 2).contiguous()   # [B,T,classes]
