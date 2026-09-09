import torch, torch.nn as nn

class FrameEncoder(nn.Module):
    """CNN appliqué frame par frame -> vecteur de features."""
    def __init__(self, in_ch=2, feat=128):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.net = nn.Sequential(
            blk(in_ch, 32),   # 128 -> 64
            blk(32, 64),      # 64  -> 32
            blk(64, 128),     # 32  -> 16
            blk(128, feat),   # 16  -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):                 # [N, 2, 128, 128]
        return self.pool(self.net(x)).flatten(1)   # [N, feat]

class RNNChain(nn.Module):
    def __init__(self, num_classes=27, feat=128, hidden=256, layers=2, rnn="gru", drop=0.3):
        super().__init__()
        self.enc = FrameEncoder(2, feat)
        R = nn.GRU if rnn.lower() == "gru" else nn.LSTM
        self.rnn = R(feat, hidden, num_layers=layers, batch_first=True,
                     dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, num_classes))
    def forward(self, x):                 # [B, T, 2, 128, 128]
        B, T = x.shape[0], x.shape[1]
        f = self.enc(x.reshape(B * T, *x.shape[2:])).reshape(B, T, -1)  # [B, T, feat]
        out, _ = self.rnn(f)
        return self.head(out[:, -1, :])   # dernier pas de temps
    def reset(self):                      # compat
        pass
