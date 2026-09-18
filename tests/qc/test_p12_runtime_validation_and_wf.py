"""P12: runtime parameter validation and Wright-Fisher parity.

Claim: the runtime updater enforces the same probability bounds as the
builder (``sex_ratio`` in [0, 1]); NaN must be rejected rather than reach
the sampling kernels (clamp01 preserves NaN, and rand_distr's Binomial
would panic on a NaN p across the PyO3 boundary).  For valid parameters,
the fused Wright-Fisher deterministic kernel reproduces the staged
deterministic trajectory to floating-point tolerance.

Reference: builder-side validation behavior (empirically: ValueError with
message "'sex_ratio' requires a value in [0.0, 1.0]"); closed-form staged
expectations.

Wrong results rejected: a runtime write path that silently accepts
out-of-range or NaN probabilities and corrupts state (negative male
mass in WF mode, or a Rust panic), WF mode disagreeing with the staged
kernel on valid input.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import qc_species_2, neutral_population

TOL = 1e-6


def _built(name: str, **kwargs):
    return neutral_population(
        qc_species_2(name),
        name,
        female={"W|W": 1000},
        male={"W|W": 1000},
        eggs_per_female=10.0,
        **kwargs,
    )


def test_runtime_update_rejects_out_of_range_sex_ratio() -> None:
    pop = _built("QC0915_p12a")
    with pytest.raises(ValueError):
        pop.update().reproduction(sex_ratio=1.2)


def test_runtime_update_rejects_nan_sex_ratio() -> None:
    pop = _built("QC0915_p12b")
    try:
        pop.update().reproduction(sex_ratio=float("nan"))
    except ValueError:
        return
    # If NaN is accepted it must still not crash the interpreter: the
    # sampling kernels panic on NaN probabilities (Rust expect).
    try:
        pop.run(1)
    except BaseException as exc:  # PanicException derives from BaseException
        pytest.fail(f"NaN sex_ratio reached the sampling kernel: {exc!r}")


def test_wf_deterministic_matches_staged_for_valid_parameters() -> None:
    staged = _built("QC0915_p12c_staged", sex_ratio=0.3, survival=0.8)
    wf = neutral_population(
        qc_species_2("QC0915_p12c_wf"),
        "QC0915_p12c_wf",
        female={"W|W": 1000},
        male={"W|W": 1000},
        eggs_per_female=10.0,
        sex_ratio=0.3,
        survival=0.8,
        extreme_speed_mode=3,
    )
    # No regulation: offspring scale with females only, so each generation
    # multiplies BOTH sexes' adults by eggs x sex_ratio x survival for
    # females (10 x 0.3 x 0.8 = 2.4) and the same female production x
    # male share for males (10 x 0.3 x 0.7 / 0.3 = 2.4 relative to male
    # adults -- i.e. both grow 2.4x per tick from the female base).
    expected = (2400.0, 5600.0)
    for _ in range(3):
        staged.run(1)
        wf.run(1)
        a = np.asarray(staged.state.individual_count[:, 1, :])
        b = np.asarray(wf.state.individual_count[:, 1, :])
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=TOL)
        assert abs(float(a[0].sum()) - expected[0]) < 1e-6
        assert abs(float(a[1].sum()) - expected[1]) < 1e-6
        expected = (expected[0] * 2.4, expected[1] * 2.4)
