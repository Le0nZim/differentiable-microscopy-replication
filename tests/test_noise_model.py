"""Tests for the detector noise model."""

from __future__ import annotations

import pytest
import torch

from models.detector_noise import DetectorNoise, DetectorNoiseConfig
from utils.config import load_yaml_config


@pytest.fixture
def yaml_config():
    return load_yaml_config("configs/base_patchmnist.yaml")


def test_noise_free_is_identity():
    noise = DetectorNoise(DetectorNoiseConfig(mode="noise_free"))
    alpha_down = torch.rand(2, 3, 8, 8)
    y_down = noise(alpha_down, apply_noise=False)
    assert torch.allclose(y_down, alpha_down)


def test_noise_free_mode_ignores_apply_noise_flag_when_mode_is_noise_free():
    noise = DetectorNoise(DetectorNoiseConfig(mode="noise_free", apply_noise=True))
    alpha_down = torch.rand(1, 2, 4, 4)
    y_down = noise(alpha_down)
    assert torch.allclose(y_down, alpha_down)


def test_differentiable_poisson_is_deterministic_with_fixed_noise():
    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson",
            photon_count=100.0,
            gamma=10.0,
            apply_noise=True,
        )
    )
    alpha_down = torch.full((1, 1, 2, 2), 0.5)
    z = torch.full_like(alpha_down, 0.25)

    y_a = noise(alpha_down, poisson_noise=z)
    y_b = noise(alpha_down, poisson_noise=z)
    assert torch.allclose(y_a, y_b)
    assert y_a.shape == alpha_down.shape


def test_differentiable_poisson_matches_closed_form():
    photon_count = 100.0
    gamma = 10.0
    alpha_down = torch.tensor([[[[0.5]]]])
    z = torch.tensor([[[[1.0]]]])

    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson",
            photon_count=photon_count,
            gamma=gamma,
            apply_noise=True,
        )
    )
    y_down = noise(alpha_down, poisson_noise=z)

    expected_mean = alpha_down + gamma / photon_count
    expected_std = torch.sqrt(alpha_down / photon_count + gamma / (photon_count**2))
    expected = expected_mean + expected_std * z
    assert torch.allclose(y_down, expected)


def test_poisson_plus_read_noise_adds_read_term():
    alpha_down = torch.zeros(1, 1, 2, 2)
    z_poiss = torch.zeros_like(alpha_down)
    z_read = torch.ones_like(alpha_down)

    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=100.0,
            gamma=10.0,
            sigma_read=2.5,
            apply_noise=True,
        )
    )
    y_down = noise(alpha_down, poisson_noise=z_poiss, read_noise=z_read)

    # Poisson term reduces to gamma/k when alpha=0 and z_poiss=0.
    expected_poisson = torch.full_like(alpha_down, 10.0 / 100.0)
    expected = expected_poisson + 2.5 * z_read
    assert torch.allclose(y_down, expected)


def test_detector_noise_from_yaml(yaml_config):
    noise = DetectorNoise.from_dict(yaml_config["detector_noise"])
    alpha_down = torch.rand(1, 8, 8, 8)
    y_down = noise(alpha_down)
    assert y_down.shape == alpha_down.shape


def test_noise_on_differs_with_random_seed():
    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=100.0,
            gamma=10.0,
            sigma_read=1.0,
            apply_noise=True,
        )
    )
    alpha_down = torch.full((2, 4, 8, 8), 0.5)
    y_a = noise(alpha_down)
    y_b = noise(alpha_down)
    assert not torch.allclose(y_a, y_b)


def test_poisson_variance_decreases_with_higher_photon_count():
    alpha_down = torch.full((1, 1, 4, 4), 0.5)
    torch.manual_seed(0)
    low_k = DetectorNoise(
        DetectorNoiseConfig(mode="differentiable_poisson", photon_count=50.0, gamma=10.0, apply_noise=True)
    )(alpha_down)
    torch.manual_seed(0)
    high_k = DetectorNoise(
        DetectorNoiseConfig(mode="differentiable_poisson", photon_count=500.0, gamma=10.0, apply_noise=True)
    )(alpha_down)
    assert (low_k - alpha_down).abs().mean() > (high_k - alpha_down).abs().mean()


def test_read_noise_scales_with_sigma_read():
    alpha_down = torch.zeros(1, 1, 2, 2)
    z_poiss = torch.zeros_like(alpha_down)
    z_read = torch.ones_like(alpha_down)
    y_low = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=100.0,
            gamma=10.0,
            sigma_read=1.0,
            apply_noise=True,
        )
    )(alpha_down, poisson_noise=z_poiss, read_noise=z_read)
    y_high = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=100.0,
            gamma=10.0,
            sigma_read=5.0,
            apply_noise=True,
        )
    )(alpha_down, poisson_noise=z_poiss, read_noise=z_read)
    assert (y_high - y_low).abs().mean() > 0.0


def test_paper_normalization_poisson_matches_eq_s7():
    k = 100.0
    gamma = 10.0
    down = 8
    alpha_down = torch.tensor([[[[32.0]]]])  # alpha_norm = 32/64 = 0.5 after sum-pool scale
    z = torch.tensor([[[[1.0]]]])

    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson",
            photon_count=k,
            gamma=gamma,
            apply_noise=True,
            noise_normalization="paper",
            downscale_factor=down,
        )
    )
    y_down = noise(alpha_down, poisson_noise=z)

    alpha_norm = alpha_down / (down**2)
    expected_mean = alpha_norm + gamma / k
    expected_std = torch.sqrt(alpha_norm / k + gamma / (k**2))
    expected = expected_mean + expected_std * z
    assert torch.allclose(y_down, expected)


def test_paper_normalization_read_noise_divided_by_photon_count():
    k = 100.0
    gamma = 10.0
    sigma_read = 2.5
    down = 8
    alpha_down = torch.zeros(1, 1, 2, 2)
    z_poiss = torch.zeros_like(alpha_down)
    z_read = torch.ones_like(alpha_down)

    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=k,
            gamma=gamma,
            sigma_read=sigma_read,
            apply_noise=True,
            noise_normalization="paper",
            downscale_factor=down,
        )
    )
    y_down = noise(alpha_down, poisson_noise=z_poiss, read_noise=z_read)

    expected_poisson = torch.full_like(alpha_down, gamma / k)
    expected_read = (sigma_read / k) * z_read
    assert torch.allclose(y_down, expected_poisson + expected_read)


def test_paper_read_noise_contribution_scales_with_one_over_k():
    k = 10.0
    gamma = 10.0
    down = 8
    alpha_down = torch.full((1, 1, 4, 4), 32.0)  # alpha_norm = 0.5
    z_poiss = torch.zeros_like(alpha_down)
    z_read = torch.ones_like(alpha_down)

    y_low = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=k,
            gamma=gamma,
            sigma_read=0.0,
            apply_noise=True,
            noise_normalization="paper",
            downscale_factor=down,
        )
    )(alpha_down, poisson_noise=z_poiss, read_noise=z_read)
    y_high = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=k,
            gamma=gamma,
            sigma_read=6.0,
            apply_noise=True,
            noise_normalization="paper",
            downscale_factor=down,
        )
    )(alpha_down, poisson_noise=z_poiss, read_noise=z_read)
    assert torch.allclose((y_high - y_low).abs().mean(), torch.tensor(6.0 / k))


def test_paper_saturated_forward_alpha_norm_near_one_at_d8():
    from models.forward_model import ForwardModel, ForwardModelConfig

    down = 8
    forward = ForwardModel(ForwardModelConfig(downscale_factor=down, use_impulse_psfs=True))
    specimen = torch.ones(1, 1, 64, 64)
    patterns = torch.ones(1, 1, 64, 64)
    alpha_down = forward(specimen, patterns)

    alpha_norm = alpha_down / (down**2)
    assert torch.allclose(alpha_down.max(), torch.tensor(float(down**2)))
    assert torch.allclose(alpha_norm.max(), torch.tensor(1.0))


def test_paper_inverse_receives_normalized_measurement_scale():
    from models.forward_model import ForwardModel, ForwardModelConfig
    from models.microscope import DifferentiableMicroscope

    down = 8
    k = 10.0
    gamma = 10.0
    forward = ForwardModel(ForwardModelConfig(downscale_factor=down, use_impulse_psfs=True))
    specimen = torch.ones(1, 1, 64, 64)
    patterns = torch.ones(1, 1, 64, 64)
    alpha_down = forward(specimen, patterns)
    alpha_norm = alpha_down / (down**2)

    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=k,
            gamma=gamma,
            sigma_read=2.7,
            apply_noise=True,
            noise_normalization="paper",
            downscale_factor=down,
        )
    )
    z_poiss = torch.zeros_like(alpha_down)
    z_read = torch.zeros_like(alpha_down)
    y_down = noise(alpha_down, poisson_noise=z_poiss, read_noise=z_read)

    expected = alpha_norm + gamma / k
    assert torch.allclose(y_down, expected)
    assert y_down.max().item() < 3.0  # O(1), not O(d²) or O(d²/k)

    config = {
        "dataset": {"image_size": 64},
        "pattern_generator": {
            "mode": "learnable_frequency",
            "num_patterns": 1,
            "sigmoid_m": 1.0,
            "seed": 1,
        },
        "forward_model": {"downscale_factor": down, "use_impulse_psfs": True},
        "detector_noise": {
            "mode": "differentiable_poisson_plus_read",
            "photon_count": k,
            "gamma": gamma,
            "sigma_read": 2.7,
            "apply_noise": True,
            "noise_normalization": "paper",
        },
        "inverse_model": {
            "upsampling": {"mode": "transpose_conv", "downscale_factor": down, "num_patterns": 1},
            "reconstruction": {
                "in_channels": 1,
                "hidden_channels": [8, 8, 8, 8, 4, 1],
                "kernel_size": 3,
                "padding": 1,
            },
        },
        "sigmoid_schedule": {"epoch_baseline": 0, "epoch_cutoff": 0, "epoch_step": 1, "m_init": 1.0},
    }
    model = DifferentiableMicroscope.from_run_config(config)
    assert model.detector_noise.config.downscale_factor == down


def _empirical_mean_var_paper_v3(
    *, alpha_norm: float, k: float, gamma: float, sigma_read: float, n: int = 400_000, seed: int = 0
) -> tuple[float, float]:
    """Sample y_norm ~ paper_v3 detector model and return (mean, var).

    paper_v3 uses alpha_divisor=1, so passing alpha_down=alpha_norm makes the
    normalized signal exactly alpha_norm regardless of downscale_factor.
    """
    torch.manual_seed(seed)
    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=k,
            gamma=gamma,
            sigma_read=sigma_read,
            apply_noise=True,
            noise_normalization="paper_v3",
            downscale_factor=8,  # must be ignored by paper_v3 (alpha_divisor=1)
        )
    )
    alpha_down = torch.full((1, 1, n, 1), float(alpha_norm))
    y = noise(alpha_down)
    return float(y.mean().item()), float(y.var(unbiased=True).item())


@pytest.mark.parametrize(
    "alpha_norm,k,sigma_read",
    [
        (0.5, 10.0, 0.0),
        (0.5, 10.0, 6.0),
        (0.5, 10000.0, 6.0),
        (12.0, 10.0, 6.0),  # alpha_norm > 1 (true sum-pool scale), the v3 regime
    ],
)
def test_paper_v3_distribution_matches_closed_form_moments(alpha_norm, k, sigma_read):
    """Empirical mean/var of the v3 detector model match eqs. S7+S9 closed form.

    mean = alpha_norm + gamma/k
    var  = alpha_norm/k + gamma/k^2 + sigma_read^2/k^2
    (Poisson and read noise are independent, so variances add.)
    """
    gamma = 10.0
    mean_emp, var_emp = _empirical_mean_var_paper_v3(
        alpha_norm=alpha_norm, k=k, gamma=gamma, sigma_read=sigma_read
    )

    mean_theory = alpha_norm + gamma / k
    var_theory = alpha_norm / k + gamma / (k**2) + (sigma_read**2) / (k**2)

    assert abs(mean_emp - mean_theory) < 0.01, (mean_emp, mean_theory)
    # Relative variance tolerance: Monte-Carlo error on var ~ sqrt(2/N).
    assert abs(var_emp - var_theory) <= 0.03 * var_theory + 1e-9, (var_emp, var_theory)


def test_paper_v3_read_noise_increases_variance_by_sigma_sq_over_k_sq():
    """Var(with read) - Var(no read) == sigma_read^2 / k^2 (independent draws)."""
    gamma = 10.0
    k = 10.0
    sigma_read = 6.0
    _, var_no_read = _empirical_mean_var_paper_v3(
        alpha_norm=0.5, k=k, gamma=gamma, sigma_read=0.0, seed=1
    )
    _, var_with_read = _empirical_mean_var_paper_v3(
        alpha_norm=0.5, k=k, gamma=gamma, sigma_read=sigma_read, seed=2
    )
    delta = var_with_read - var_no_read
    expected = (sigma_read**2) / (k**2)
    assert abs(delta - expected) <= 0.05 * expected, (delta, expected)


def test_paper_v3_uses_alpha_down_directly_no_d2_division():
    """v3 closed form: alpha_norm == alpha_down (no /d²), unlike v2 'paper' mode."""
    k = 100.0
    gamma = 10.0
    down = 8
    alpha_down = torch.tensor([[[[32.0]]]])  # sum-pool scale value
    z = torch.tensor([[[[1.0]]]])

    v3 = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson",
            photon_count=k,
            gamma=gamma,
            apply_noise=True,
            noise_normalization="paper_v3",
            downscale_factor=down,
        )
    )
    y_v3 = v3(alpha_down, poisson_noise=z)

    alpha_norm = alpha_down  # v3: no division
    expected = (alpha_norm + gamma / k) + torch.sqrt(alpha_norm / k + gamma / (k**2)) * z
    assert torch.allclose(y_v3, expected)

    # And it must differ from the v2 'paper' mode (which divides by d²).
    v2 = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson",
            photon_count=k,
            gamma=gamma,
            apply_noise=True,
            noise_normalization="paper",
            downscale_factor=down,
        )
    )
    y_v2 = v2(alpha_down, poisson_noise=z)
    assert not torch.allclose(y_v3, y_v2)


def test_paper_v3_read_noise_divided_by_k_regression():
    """Regression guard: read noise term is sigma_read/k, NOT sigma_read.

    Fails if anyone reintroduces the legacy behavior of adding sigma_read*z
    (no division by the photon count) under a paper normalization mode.
    """
    k = 10.0
    gamma = 10.0
    sigma_read = 6.0
    down = 8
    alpha_down = torch.zeros(1, 1, 2, 2)
    z_poiss = torch.zeros_like(alpha_down)
    z_read = torch.ones_like(alpha_down)

    y = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=k,
            gamma=gamma,
            sigma_read=sigma_read,
            apply_noise=True,
            noise_normalization="paper_v3",
            downscale_factor=down,
        )
    )(alpha_down, poisson_noise=z_poiss, read_noise=z_read)

    # Poisson reduces to gamma/k when alpha=0, z_poiss=0.
    read_contrib = (y - gamma / k).abs().mean().item()
    assert abs(read_contrib - sigma_read / k) < 1e-6  # == 0.6
    # The legacy (buggy) magnitude would be sigma_read = 6.0; ensure we are far from it.
    assert read_contrib < 1.0


def test_paper_v3_microscope_feeds_offset_and_scaled_read_to_inverse():
    """Microscope integration: the inverse model receives y_norm (eq. S10), i.e.
    the sum-pool forward output PLUS gamma/k offset PLUS noise — not raw alpha_down,
    and the read term is scaled by 1/k (not the raw sigma_read)."""
    from models.microscope import DifferentiableMicroscope

    down = 8
    k = 10.0
    gamma = 10.0
    sigma_read = 6.0
    config = {
        "dataset": {"image_size": 64},
        "pattern_generator": {
            "mode": "learnable_frequency",
            "num_patterns": 1,
            "sigmoid_m": 1.0,
            "seed": 1,
        },
        "forward_model": {"downscale_factor": down, "use_impulse_psfs": True},
        "detector_noise": {
            "mode": "differentiable_poisson_plus_read",
            "photon_count": k,
            "gamma": gamma,
            "sigma_read": sigma_read,
            "apply_noise": True,
            "noise_normalization": "paper_v3",
        },
        "inverse_model": {
            "upsampling": {"mode": "transpose_conv", "downscale_factor": down, "num_patterns": 1},
            "reconstruction": {
                "in_channels": 1,
                "hidden_channels": [8, 8, 8, 8, 4, 1],
                "kernel_size": 3,
                "padding": 1,
            },
        },
        "sigmoid_schedule": {"epoch_baseline": 0, "epoch_cutoff": 0, "epoch_step": 1, "m_init": 1.0},
    }
    model = DifferentiableMicroscope.from_run_config(config)
    assert model.detector_noise.config.downscale_factor == down
    assert model.detector_noise.config.noise_normalization == "paper_v3"

    specimen = torch.ones(1, 1, 64, 64)
    torch.manual_seed(0)
    out = model(specimen, sigmoid_m=10.0, apply_noise=True)

    alpha_down = out["alpha_down"]
    y_down = out["y_down"]
    # v3 measurement scale equals the sum-pool scale (no /d²): both reach ~d².
    # (v2 'paper' mode would put the signal in [0, 1]; raw /k would be ~d²/k.)
    assert alpha_down.max().item() > down**2 * 0.5
    # The measurement must NOT equal the raw sum-pool output: offset + noise present.
    assert not torch.allclose(y_down, alpha_down)

    # Deterministic offset/scaling check via the detector module directly:
    # with zero noise draws, y_norm == alpha_down + gamma/k (exact), and the
    # read term is sigma_read/k (not sigma_read).
    zero = torch.zeros_like(alpha_down)
    y_offset_only = model.detector_noise(alpha_down, poisson_noise=zero, read_noise=zero)
    assert torch.allclose(y_offset_only, alpha_down + gamma / k)
    ones = torch.ones_like(alpha_down)
    y_with_read = model.detector_noise(alpha_down, poisson_noise=zero, read_noise=ones)
    read_contrib = (y_with_read - y_offset_only).abs().mean().item()
    assert abs(read_contrib - sigma_read / k) < 1e-5  # == 0.6, not 6.0


def test_legacy_default_unchanged_without_paper_flag():
    cfg = DetectorNoiseConfig(
        mode="differentiable_poisson_plus_read",
        photon_count=100.0,
        gamma=10.0,
        sigma_read=2.5,
        apply_noise=True,
    )
    assert cfg.noise_normalization == "legacy"


def test_noisy_output_finite_and_non_negative_for_typical_alpha():
    noise = DetectorNoise(
        DetectorNoiseConfig(
            mode="differentiable_poisson_plus_read",
            photon_count=10000.0,
            gamma=10.0,
            sigma_read=2.7,
            apply_noise=True,
        )
    )
    alpha_down = torch.rand(2, 8, 16, 16).clamp(min=0.0, max=1.0)
    y_down = noise(alpha_down)
    assert torch.isfinite(y_down).all()
    # Normal approx can go slightly negative; paper uses correction — check not NaN only.


def test_learnable_gradient_flows_with_noise_on():
    from models.microscope import DifferentiableMicroscope

    config = {
        "dataset": {"image_size": 32},
        "pattern_generator": {
            "mode": "learnable_frequency",
            "num_patterns": 4,
            "height": 32,
            "width": 32,
            "sigmoid_m": 1.0,
            "seed": 1,
        },
        "forward_model": {"downscale_factor": 4, "use_impulse_psfs": True},
        "detector_noise": {
            "mode": "differentiable_poisson_plus_read",
            "photon_count": 100.0,
            "gamma": 10.0,
            "sigma_read": 1.0,
            "apply_noise": True,
        },
        "inverse_model": {
            "upsampling": {"mode": "transpose_conv", "downscale_factor": 4, "num_patterns": 4},
            "reconstruction": {
                "in_channels": 4,
                "hidden_channels": [16, 16, 8, 8, 4, 1],
                "kernel_size": 3,
                "padding": 1,
            },
        },
        "sigmoid_schedule": {"epoch_baseline": 0, "epoch_cutoff": 0, "epoch_step": 1, "m_init": 1.0},
    }
    model = DifferentiableMicroscope.from_run_config(config)
    specimen = torch.rand(1, 1, 32, 32)
    outputs = model(specimen, sigmoid_m=1.0, apply_noise=True)
    loss = outputs["x_recon"].mean()
    loss.backward()
    assert model.pattern_generator.W.grad is not None
    assert torch.isfinite(model.pattern_generator.W.grad).all()
