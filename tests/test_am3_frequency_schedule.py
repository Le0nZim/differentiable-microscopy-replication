"""AM-3 Phase 3: frequency-domain optimization + custom-sigmoid schedule audit.

Verifies the differentiable illumination parameterization matches the paper:
  * eq. 6:  tau = real(ifft(W))
  * eq. 7:  sigmoid-friendly init  W ~ F(tau_0),  tau_0 ~ N(0, 1)
  * eq. 4/5: H_t = sigmoid(m * tau)  (custom sigmoid with sharpness m)
  * Algorithm 1: inverse-only warmup, then m increases on a step schedule.
  * variant D (learnable_spatial) removes the frequency-domain parameterization
    but keeps H_t learnable end-to-end.
"""

from __future__ import annotations

import torch

from models.pattern_generator import (
    PatternGenerator,
    PatternGeneratorConfig,
    SigmoidSchedule,
)


def _make(mode: str, seed: int = 0) -> PatternGenerator:
    return PatternGenerator(
        PatternGeneratorConfig(mode=mode, num_patterns=4, height=32, width=32, seed=seed)
    )


def test_frequency_tau_is_real_ifft_of_W():
    pg = _make("learnable_frequency")
    tau = pg._spatial_tau()
    expected = torch.fft.ifft2(pg.W).real
    assert torch.allclose(tau, expected)
    assert not torch.is_complex(tau)


def test_sigmoid_friendly_init_roundtrip_is_standard_normal():
    """W = fft(tau_0), tau_0 ~ N(0,1)  =>  real(ifft(W)) == tau_0 ~ N(0,1)."""
    pg = _make("learnable_frequency", seed=123)
    tau0 = pg._spatial_tau()  # equals the tau_0 used to build W at init
    # Should be (approximately) standard normal, NOT tiny constants.
    assert abs(float(tau0.mean().item())) < 0.1
    assert 0.7 < float(tau0.std().item()) < 1.3
    # Sanity: feeding tau0 into custom sigmoid at m=1 gives a usable spread (not all ~0.5).
    h = torch.sigmoid(1.0 * tau0)
    assert float(h.std().item()) > 0.1


def test_custom_sigmoid_uses_m():
    pg = _make("learnable_frequency")
    tau = pg._spatial_tau()
    for m in (1.0, 4.0, 8.0):
        h = pg.forward(sigmoid_m=m)
        assert torch.allclose(h, torch.sigmoid(m * tau), atol=1e-6)
    # higher m => sharper (more values pushed toward {0,1})
    h1 = pg.forward(sigmoid_m=1.0)
    h8 = pg.forward(sigmoid_m=8.0)
    bf = lambda x: float(((x < 0.1) | (x > 0.9)).float().mean().item())
    assert bf(h8) > bf(h1)


def test_learnable_spatial_has_no_frequency_W():
    pg = _make("learnable_spatial")
    assert pg.W is None
    assert pg.tau is not None
    h = pg.forward(sigmoid_m=2.0)
    assert torch.allclose(h, torch.sigmoid(2.0 * pg.tau), atol=1e-6)


def test_frequency_W_receives_gradient():
    pg = _make("learnable_frequency")
    h = pg.forward(sigmoid_m=1.0)
    h.sum().backward()
    assert pg.W.grad is not None
    assert torch.isfinite(pg.W.grad).all()
    assert float(pg.W.grad.abs().sum().item()) > 0


def test_schedule_warmup_then_harden():
    sched = SigmoidSchedule(epoch_baseline=10, epoch_cutoff=20, epoch_step=5, m_init=1.0)
    # During warmup: illumination frozen, m stays at init.
    assert sched.should_freeze_patterns(5) is True
    assert sched.step(5) == 1.0
    assert sched.should_freeze_patterns(10) is True
    # After baseline but before cutoff: not frozen, m stays at 1.
    assert sched.should_freeze_patterns(11) is False
    assert sched.step(15) == 1.0
    # After cutoff, on step boundaries m increments.
    sched.reset()
    m_at_25 = sched.step(25)  # 25 > 20 and 25 % 5 == 0 => +1
    assert m_at_25 == 2.0
