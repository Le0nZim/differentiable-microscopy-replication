"""Evaluation helpers must not change a model's training state or BN buffers."""
from functools import wraps
from contextlib import contextmanager

import torch


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
