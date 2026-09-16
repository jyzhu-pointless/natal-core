"""Analytical checks for the proposed density-hook demo, not engine parity.

References use an exact two-age recurrence: births = adult females * eggs,
next adults = births * density multiplier * baseline survival. Relative
tolerance 1e-12 covers rounding in a few float64 reductions and divisions.
"""

from functools import partial
import runpy
from pathlib import Path

import numpy as np
import pytest

from demos.density_regulation import (
    Demography,
    DensityHook,
    WeightedPressure,
    advance,
    beverton_holt,
    density_multiplier,
)


@pytest.mark.parametrize("age, expected_pressure", [(0, 1000.0), (1, 200.0)])
def test_reference_phase_and_fixed_point(age: int, expected_pressure: float) -> None:
    """A pre-birth age-0 zero must become 1000 births before C* is measured."""
    models = {"A": Demography(10)}
    reference = {"A": np.array([[0.0, 100.0], [0.0, 100.0]])}
    hook = DensityHook("A", WeightedPressure(age, {"A": 1.0}))
    calibration = hook.calibrate(models, reference)
    assert calibration.reference_pressure == expected_pressure
    # 100 females * 10 eggs * 0.5 survival = 500 unregulated recruits.
    assert calibration.reference_multiplier == 0.4
    result, _ = advance(models, reference, reference, [hook])
    np.testing.assert_allclose(result["A"], reference["A"], rtol=1e-12, atol=0)


@pytest.mark.parametrize("adults, expected", [(0.0, 0.0), (100.0, 400 / 3), (400.0, 800 / 3)])
def test_off_equilibrium_recurrence(adults: float, expected: float) -> None:
    """For K=200 and r=2 the recurrence is N'=2N/(1+N/200)."""
    model = Demography(10)
    reference = {"A": model.reference_distribution(200)}
    counts = {"A": reference["A"] * (adults / 200)}
    result, _ = advance({"A": model}, counts, reference, [DensityHook("A", WeightedPressure(1, {"A": 1.0}))])
    assert result["A"][:, 1].sum() == pytest.approx(expected, rel=1e-12, abs=0)
    np.testing.assert_array_equal(counts["A"][:, 1], [adults / 2, adults / 2])
    np.testing.assert_array_equal(reference["A"][:, 1], [100, 100])


def test_adult_and_newborn_pressure_distinguish_a_birth_pulse() -> None:
    """A temporary birth pulse changes newborn pressure, not adult pressure."""
    model = Demography(10)
    reference = {"A": model.reference_distribution(200)}
    actual = {"A": model.reproduce(reference["A"])}
    actual["A"][:, 0] *= 2
    factors = []
    for age in (0, 1):
        hook = DensityHook("A", WeightedPressure(age, {"A": 1.0}))
        calibration = hook.calibrate({"A": model}, reference)
        factors.append(density_multiplier(hook.pressure(actual), calibration.reference_pressure, calibration.reference_multiplier, response=hook.response))
    np.testing.assert_allclose(factors, [4 / 15, 0.4], rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "age,expected_a,expected_b",
    [(1, 2000 / 11, 1800 / 13), (0, 2400 / 13, 1560 / 11)],
)
def test_two_species_use_one_unmodified_pressure_snapshot(
    age: int, expected_a: float, expected_b: float,
) -> None:
    """Check coupled recruitment without allowing earlier scaling to alter pressure.

    With newborn pressure, reference births A=1000,B=400 become A=1000,B=800.
    Thus the pressure ratios are 1400/1200 and 1050/650, giving response
    factors 12/13 and 39/55. Adult pressure gives factors 10/11 and 9/13.
    """
    models = {"A": Demography(10), "B": Demography(8, (0.4, 0.4))}
    reference = {name: model.reference_distribution(total) for (name, model), total in zip(models.items(), (200, 100))}
    hooks = [
        DensityHook("A", WeightedPressure(age, {"A": 1, "B": 0.5})),
        DensityHook("B", WeightedPressure(age, {"A": 0.25, "B": 1}), partial(beverton_holt, r=3)),
    ]
    # Each species has its own calibration, even at joint equilibrium.
    equilibrium, _ = advance(models, reference, reference, hooks)
    for name in reference:
        np.testing.assert_allclose(equilibrium[name], reference[name], rtol=1e-12, atol=0)
    actual = {"A": reference["A"], "B": reference["B"] * 2}
    result, _ = advance(models, actual, reference, hooks)
    reverse, _ = advance(models, actual, reference, hooks[::-1])
    assert result["A"][:, 1].sum() == pytest.approx(expected_a, rel=1e-12)
    assert result["B"][:, 1].sum() == pytest.approx(expected_b, rel=1e-12)
    for name in reference:
        np.testing.assert_array_equal(result[name], reverse[name])


def test_demography_change_recalibrates_without_moving_reference() -> None:
    """Doubling eggs halves m* while preserving the supplied target distribution."""
    old = Demography(10)
    new = Demography(20)
    reference = {"A": old.reference_distribution(200)}
    hook = DensityHook("A", WeightedPressure(1, {"A": 1}))
    assert hook.calibrate({"A": old}, reference).reference_multiplier == 0.4
    assert hook.calibrate({"A": new}, reference).reference_multiplier == 0.2
    result, _ = advance({"A": new}, reference, reference, [hook])
    np.testing.assert_allclose(result["A"], reference["A"], rtol=1e-12, atol=0)


def test_reference_uses_surviving_sex_ratio_and_allows_multiplier_above_one() -> None:
    """Birth ratio 1:1 with survival 0.25:0.75 needs an age-1 ratio of 1:3."""
    model = Demography(2, (0.25, 0.75))
    reference = {"A": model.reference_distribution(200)}
    np.testing.assert_array_equal(reference["A"], [[0, 50], [0, 150]])
    hook = DensityHook("A", WeightedPressure(1, {"A": 1}))
    assert hook.calibrate({"A": model}, reference).reference_multiplier == 4
    result, _ = advance({"A": model}, reference, reference, [hook])
    np.testing.assert_array_equal(result["A"], reference["A"])


@pytest.mark.parametrize("values", [(float("nan"), 1, 1), (-1, 1, 1), (1, 0, 1), (1, 1, -1), (1e308, 1e-308, 1)])
def test_invalid_density_inputs(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        density_multiplier(*values, response=beverton_holt)


@pytest.mark.parametrize("factor", [-1.0, float("inf"), float("nan")])
def test_invalid_response_output(factor: float) -> None:
    with pytest.raises(ValueError):
        density_multiplier(1, 1, 1, response=lambda _: factor)


@pytest.mark.parametrize("x,r", [(-1, 2), (1, 0.5), (float("inf"), 2)])
def test_invalid_response_domain(x: float, r: float) -> None:
    with pytest.raises(ValueError):
        beverton_holt(x, r=r)


@pytest.mark.parametrize("eggs,survival,female", [(0, (0.5, 0.5), 0.5), (10, (0, 1), 0.5), (10, (1, 1), 1), (float("nan"), (1, 1), 0.5)])
def test_invalid_demography(eggs, survival, female) -> None:
    with pytest.raises(ValueError):
        Demography(eggs, survival, female)


@pytest.mark.parametrize("total", [0, -1, float("inf")])
def test_invalid_reference_total(total: float) -> None:
    with pytest.raises(ValueError):
        Demography(10).reference_distribution(total)


@pytest.mark.parametrize("counts", [np.zeros(2), np.full((2, 2), -1), np.full((2, 2), float("nan")), np.ones((2, 2))])
def test_invalid_pre_reproduction_counts(counts) -> None:
    with pytest.raises(ValueError):
        Demography(10).reproduce(counts)


@pytest.mark.parametrize("age,weights", [(2, {"A": 1}), (0, {}), (1, {"A": -1}), (1, {"A": float("inf")})])
def test_invalid_pressure_definition(age, weights) -> None:
    with pytest.raises(ValueError):
        WeightedPressure(age, weights)


def test_invalid_calibration_and_hook_configuration() -> None:
    models = {"A": Demography(10, (0.25, 0.75))}
    reference = {"A": models["A"].reference_distribution(200)}
    pressure = WeightedPressure(1, {"A": 1})
    hook = DensityHook("A", pressure)
    with pytest.raises(ValueError, match="g\\(1\\)"):
        DensityHook("A", pressure, lambda _: 2).calibrate(models, reference)
    with pytest.raises(ValueError, match="positive reference recruitment"):
        hook.calibrate(models, {"A": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="sex composition"):
        hook.calibrate(models, {"A": np.array([[0., 100.], [0., 100.]])})
    for hooks in ([], [hook, hook]):
        with pytest.raises(ValueError, match="exactly one"):
            advance(models, reference, reference, hooks)


def test_complete_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """Execute the delivered command's main path, including all printed scenarios."""
    path = Path(__file__).resolve().parents[1] / "demos" / "density_regulation.py"
    runpy.run_path(str(path), run_name="__main__")
    output = capsys.readouterr().out
    assert "Birth pulse" in output
    assert "reference adults remain 200.0" in output
