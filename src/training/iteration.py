"""Training iteration without caching augmented batches."""

from __future__ import annotations


def repeat_dataloader(loader):
    """Start a fresh loader pass after exhaustion; never retain old batches."""
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise ValueError("Cannot train with an empty dataloader")
