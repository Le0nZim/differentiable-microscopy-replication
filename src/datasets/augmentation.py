"""An advancing crop RNG independent of model initialization and detector noise."""
import torch
from torch.utils.data import get_worker_info


def training_crop_generator(dataset):
    worker = get_worker_info()
    # Worker seeds come from the dataloader generator. In the main process use
    # the dataset seed, never torch.initial_seed() / the model's global RNG.
    seed = int(dataset.config.seed) + (worker.seed if worker else 30_000)
    key = (worker.id if worker else -1, seed)
    if getattr(dataset, "_crop_rng_key", None) != key:
        dataset._crop_rng_key = key
        dataset._crop_rng = torch.Generator().manual_seed(seed % (2**63 - 1))
    return dataset._crop_rng
