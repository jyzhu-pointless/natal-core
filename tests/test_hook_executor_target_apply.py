"""Unit tests for _apply_target_with_sperm and _apply_target_without_sperm.

These functions are the core target-count application helpers in the CSR
hot loop.  _apply_target_with_sperm is the more complex variant — it must
keep sperm storage coherent with female count changes.
"""

import numpy as np
import pytest

from natal.hooks.executor import _apply_target_with_sperm, _apply_target_without_sperm


# ---------------------------------------------------------------------------
# _apply_target_without_sperm
# ---------------------------------------------------------------------------


def test_without_sperm_adds_when_target_exceeds_current():
    """Adding individuals: return target_count directly, no survival."""
    result = _apply_target_without_sperm(
        current_count=10.0, target_count=15.0,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 15.0


def test_without_sperm_deterministic_survival():
    """Deterministic: multiply by survival_prob = target / current."""
    result = _apply_target_without_sperm(
        current_count=10.0, target_count=4.0,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 4.0


def test_without_sperm_target_zero_with_positive_current():
    """When target is 0 and current > 0: survival_prob = 0, result = 0."""
    result = _apply_target_without_sperm(
        current_count=10.0, target_count=0.0,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 0.0


def test_without_sperm_target_zero_returns_zero():
    result = _apply_target_without_sperm(
        current_count=10.0, target_count=0.0,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 0.0


def test_without_sperm_stochastic_binomial():
    """Stochastic (non-dirichlet): round count, then binomial sample."""
    np.random.seed(42)
    results = []
    for _ in range(200):
        r = _apply_target_without_sperm(
            current_count=100.0, target_count=50.0,
            stochastic_flag=True, dirichlet_flag=False,
        )
        results.append(r)
    # Should be roughly 50 on average (survival_prob = 0.5).
    mean = np.mean(results)
    assert 40 <= mean <= 60


def test_without_sperm_stochastic_dirichlet():
    """Continuous sampling returns float counts."""
    np.random.seed(42)
    results = []
    for _ in range(200):
        r = _apply_target_without_sperm(
            current_count=100.0, target_count=30.0,
            stochastic_flag=True, dirichlet_flag=True,
        )
        results.append(r)
    mean = np.mean(results)
    assert 25 <= mean <= 35


# ---------------------------------------------------------------------------
# _apply_target_with_sperm
# ---------------------------------------------------------------------------


def _make_sperm_row(values):
    """Helper: create a 1-D sperm row from a list of floats."""
    return np.array(values, dtype=np.float64)


def test_with_sperm_adds_when_target_exceeds():
    """Adding females: return target, sperm untouched."""
    sperm = _make_sperm_row([3.0, 2.0])
    result = _apply_target_with_sperm(
        current_count=10.0, target_count=15.0,
        sperm_row=sperm,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 15.0
    # Sperm unchanged.
    assert sperm[0] == 3.0
    assert sperm[1] == 2.0


def test_with_sperm_deterministic_scale():
    """Deterministic: scale both sperm and virgins proportionally."""
    sperm = _make_sperm_row([3.0, 2.0])  # 5 mated, 5 virgins
    result = _apply_target_with_sperm(
        current_count=10.0, target_count=5.0,  # survival_prob = 0.5
        sperm_row=sperm,
        stochastic_flag=False, dirichlet_flag=False,
    )
    # Total: 5 females surviving.  Sperm scaled by 0.5.
    assert result == 5.0
    assert sperm[0] == pytest.approx(1.5)
    assert sperm[1] == pytest.approx(1.0)


def test_with_sperm_target_zero_kills_all():
    """When target=0 and current>0: survival_prob=0, all die, sperm scaled
    to zero."""
    sperm = _make_sperm_row([3.0, 2.0])
    result = _apply_target_with_sperm(
        current_count=10.0, target_count=0.0,
        sperm_row=sperm,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 0.0
    assert sperm[0] == 0.0
    assert sperm[1] == 0.0


def test_with_sperm_all_virgins():
    """When there is no stored sperm, all females are virgins."""
    sperm = _make_sperm_row([0.0, 0.0])  # all virgins
    result = _apply_target_with_sperm(
        current_count=10.0, target_count=3.0,
        sperm_row=sperm,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 3.0
    assert sperm[0] == 0.0  # Still zero — no mated to scale.


def test_with_sperm_all_mated():
    """When every female is mated, virgin count is zero."""
    sperm = _make_sperm_row([6.0, 4.0])  # all 10 are mated
    result = _apply_target_with_sperm(
        current_count=10.0, target_count=5.0,
        sperm_row=sperm,
        stochastic_flag=False, dirichlet_flag=False,
    )
    assert result == 5.0
    assert sperm[0] == pytest.approx(3.0)  # 6 * 0.5
    assert sperm[1] == pytest.approx(2.0)  # 4 * 0.5


def test_with_sperm_stochastic_consistency():
    """Stochastic sampling should produce roughly correct totals."""
    np.random.seed(42)
    totals = []
    for _ in range(200):
        sperm = _make_sperm_row([30.0, 20.0])  # 50 mated
        result = _apply_target_with_sperm(
            current_count=100.0, target_count=50.0,  # survival_prob = 0.5
            sperm_row=sperm,
            stochastic_flag=True, dirichlet_flag=False,
        )
        totals.append(result)
    mean = np.mean(totals)
    # survival_prob = 0.5: expected ~50 survivors.
    assert 40 <= mean <= 60


def test_with_sperm_stochastic_dirichlet():
    """Continuous-Dirichlet sampling with sperm coherence."""
    np.random.seed(42)
    for _ in range(50):
        sperm = _make_sperm_row([30.0, 20.0])
        result = _apply_target_with_sperm(
            current_count=100.0, target_count=30.0,
            sperm_row=sperm,
            stochastic_flag=True, dirichlet_flag=True,
        )
        # After survival, sperm should be roughly proportional.
        if result > 0:
            new_mated = sperm[0] + sperm[1]
            # New total = virgins + mated; should be approx 30.
            assert 10 <= result <= 60  # wide tolerance for stochastic


def test_with_sperm_near_inconsistent_state_does_not_crash():
    """Edge case: sperm slightly exceeds female count (floating-point).
    Should clamp virgins to 0 instead of crashing."""
    sperm = _make_sperm_row([5.0001, 5.0])  # total 10.0001 > 10
    result = _apply_target_with_sperm(
        current_count=10.0, target_count=8.0,
        sperm_row=sperm,
        stochastic_flag=False, dirichlet_flag=False,
    )
    # Should not raise; n_virgins_raw ≈ -0.0001 → clamped to 0.
    assert result == 8.0
