import torch
from spikingjelly.clock_driven import functional


@torch.no_grad()
def update_bntt_running_stats(model, loader, device, num_batches=50):
    """Precise-BN : reestime running_mean/var de tous les BatchNorm (dont BNTT)
    sur num_batches lots train, en moyenne cumulative (momentum=None)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    saved = {}
    for bn in bns:
        bn.reset_running_stats()
        saved[bn] = bn.momentum
        bn.momentum = None
    model.train()
    n = 0
    for x, _ in loader:
        functional.reset_net(model)
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        model(x)
        n += 1
        if n >= num_batches:
            break
    functional.reset_net(model)
    for bn in bns:
        bn.momentum = saved[bn]
    return n
