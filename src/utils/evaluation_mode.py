"""Evaluation helpers must not change a model's training state or BN buffers."""
from functools import wraps
from contextlib import contextmanager

import torch


@contextmanager
def evaluation_rng(seed, device):
    """Repeat detector draws without consuming the optimizer's random stream."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.set_rng_state(torch.Generator().manual_seed(int(seed)).get_state())
            if device.type == "cuda":
                torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(int(seed)).get_state(), device)
        yield


@contextmanager
def evaluating(model):
    """Preserve even mixed train/eval submodules, including on exceptions."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def evaluation_mode(function):
    """Temporarily evaluate the first (model) argument, preserving mixed modes."""
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        with evaluating(model):
            return function(model, *args, **kwargs)
    return wrapped
