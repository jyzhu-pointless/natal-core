"""R4-12: positive controls for the input-validation net (no defect found).

Every branch below was attacked and holds:

- ``low_density_growth_rate >= 1`` is enforced for the compensatory family
  (``linear``/``beverton_holt``/``ricker``) at build time — a sub-unit r
  would make ``g(x) = r^(1-x)`` *increasing* in the competition ratio, i.e.
  density would boost rather than suppress recruitment;
- ``fixed`` accepts any r (the curve ignores it);
- r = 1 is legal (neutral curve), boundary semantics intact;
- ``sex_ratio``, ``eggs_per_female``, ``carrying_capacity`` bounds are
  enforced on the builder routes;
- ``tensor_write`` rejects negative/NaN entries for every tensor,
  including the fertility tensor whose upper bound is *not* enforced (that
  gap is reported by test_r4_11_calibration_inputs.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers_r4 import discrete_pop, species_locus


def _builder(name: str, **overrides):
    params = {
        "eggs_per_female": 10.0,
        "growth_mode": "beverton_holt",
        "carrying_capacity": 1000.0,
        "low_density_growth_rate": 3.0,
    }
    params.update(overrides)
    eggs = params.pop("eggs_per_female")
    return discrete_pop(
        name,
        species=species_locus(f"R4_12_{name}", ["W"]),
        female={"W|W": 500.0},
        male={"W|W": 500.0},
        eggs_per_female=eggs,
        survival=1.0,
        **params,
    )


@pytest.mark.parametrize("mode", ["linear", "beverton_holt", "ricker"])
def test_sub_unit_growth_rate_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        _builder(f"sub_{mode}", growth_mode=mode, low_density_growth_rate=0.5)


@pytest.mark.parametrize("mode", ["beverton_holt", "ricker"])
def test_unit_growth_rate_is_legal(mode: str) -> None:
    pop = _builder(f"unit_{mode}", growth_mode=mode, low_density_growth_rate=1.0)
    pop.run(50)
    counts = np.asarray(pop.state.individual_count)
    assert np.isfinite(counts).all()
    assert counts.sum() > 0.0


def test_fixed_mode_requires_the_growth_rate_lower_bound() -> None:
    """The rate retains its intrinsic domain even when the curve ignores it."""
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        _builder("fixed_r", growth_mode="fixed", low_density_growth_rate=0.5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sex_ratio": 1.5}, "sex_ratio"),
        ({"eggs_per_female": -1.0}, "eggs_per_female"),
        ({"carrying_capacity": -5.0}, "carrying_capacity"),
    ],
)
def test_builder_routes_enforce_bounds(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _builder(f"bounds_{match}", **kwargs)


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_tensor_write_rejects_negative_or_nan(value: float) -> None:
    pop = _builder("tensor_guard")
    with pytest.raises(ValueError, match="fertility"):
        pop.params.tensor_write("fertility", np.array([0.0, value]))
