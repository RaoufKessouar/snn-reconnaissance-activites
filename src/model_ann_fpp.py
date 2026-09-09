import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, stride=1, k=3, pad=1):
        super().__init__()
        self.conv=nn.Conv2d(cin,cout,k,stride,pad,bias=False)
        self.bn=nn.BatchNorm2d(cout,eps=1e-4,momentum=0.1)
        self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, k=3):
        super().__init__()
        self.c1=ConvBNReLU(cin,cout,stride,k,k//2)
        self.c2=ConvBNReLU(cout,cout,1,k,k//2)
        self.down = stride!=1 or cin!=cout
        if self.down:
            self.dconv=nn.Conv2d(cin,cout,1,stride,bias=False)
            self.dbn=nn.BatchNorm2d(cout,eps=1e-4,momentum=0.1)
    def forward(self,x):
        idt=x; out=self.c2(self.c1(x))
        if self.down: idt=self.dbn(self.dconv(x))
        return out+idt


class ANNResNetFPP(nn.Module):
    """ANN-BN (ReLU + BatchNorm, AUCUNE dynamique temporelle).
    Chaque frame traitee independamment -> sortie PAR FRAME [B,T,num_classes]."""
    def __init__(self, in_channels=2, base=32, num_classes=9, n=6, num_steps=40, kernel_size=3):
        super().__init__()
        self.num_steps=num_steps
        self.conv_init=ConvBNReLU(in_channels,base,1,kernel_size,kernel_size//2)
        self.stage1=self._stage(base,base,n,1)
        self.stage2=self._stage(base,base*2,n,2)
        self.stage3=self._stage(base*2,base*4,n,2)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(base*4,num_classes,bias=False)
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)): nn.init.xavier_uniform_(m.weight,gain=2.0)
    def _stage(self,cin,cout,nb,st):
        blocks=[ResidualBlock(cin,cout,st)]
        for _ in range(1,nb): blocks.append(ResidualBlock(cout,cout,1))
        return nn.Sequential(*blocks)
    def forward(self,x):                     # x:[B,T,2,H,W]
        B,T=x.shape[0],x.shape[1]
        x=x.flatten(0,1)                     # [B*T,2,H,W] : chaque frame independante
        out=self.conv_init(x)
        out=self.stage1(out); out=self.stage2(out); out=self.stage3(out)
        out=torch.flatten(self.pool(out),1)  # [B*T,C]
        out=self.fc(out)                     # [B*T,num_classes]
        return out.view(B,T,-1)              # [B,T,classes]
