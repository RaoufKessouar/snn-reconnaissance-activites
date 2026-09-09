"""Utilities for recalibrating running normalisation statistics.

Only an *training* loader must be passed here.  Recalibrating on validation or
test inputs is transductive evaluation and contradicts the protocol described
in the report.
"""

from __future__ import annotations

import torch
from spikingjelly.clock_driven import functional


def _running_norms(model):
    """Return PyTorch BN and custom tdBN modules with running statistics."""
    return [
        module
        for module in model.modules()
        if hasattr(module, "running_mean")
        and hasattr(module, "running_var")
        and hasattr(module, "reset_running_stats")
        and hasattr(module, "momentum")
    ]


@torch.no_grad()
def recalibrate_running_stats(model, loader, device, num_batches=30):
    """Recompute running statistics from ``num_batches`` training batches.

    The model's training/evaluation mode and every normaliser's momentum are
    restored even if a data-loading or forward-pass error occurs.
    """
    norms = _running_norms(model)
    previous_training = model.training
    momenta = {norm: norm.momentum for norm in norms}
    processed = 0

    try:
        for norm in norms:
            norm.reset_running_stats()
            norm.momentum = None

        model.train()
        for inputs, _ in loader:
            functional.reset_net(model)
            model(inputs.to(device, dtype=torch.float32))
            processed += 1
            if processed >= num_batches:
                break
    finally:
        functional.reset_net(model)
        for norm, momentum in momenta.items():
            norm.momentum = momentum
        model.train(previous_training)

    return processed


# Backward-compatible name used by the historical experiment scripts.
update_bntt_running_stats = recalibrate_running_stats
