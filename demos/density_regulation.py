"""Executable sketch of the proposed density-regulation API, not a NATAL API.

Run: .venv/bin/python demos/density_regulation.py

The deterministic model has two sexes and two ages: newborns (0) and
reproductive adults (1). Adults die after reproducing; newborns undergo
density regulation and then sex-specific baseline survival into age 1.
There is no mating limitation, genetics, migration, or stochastic sampling.
Reference distributions describe the BEFORE-reproduction boundary, where
age 0 is empty. Calibration evaluates pressure AFTER reference reproduction,
at the same boundary used by the density hooks on the actual population.

This sketch supports positive reference populations with positive egg
production and survival. Zero habitat/extinction calibration is deliberately
not defined here. Multipliers are not probabilities and are not capped at 1.
The response formulas mirror rust/src/kernels/density_regulation.rs; that
module currently has no standalone Python curve exports. The existing FIXED
hard-cap model has a different calibration and is not represented here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray

Counts = NDArray[np.float64]
Community = Mapping[str, Counts]
Response = Callable[[float], float]
Pressure = Callable[[Community], float]


def density_multiplier(
    pressure: float,
    reference_pressure: float,
    reference_multiplier: float,
    *,
    response: Response,
) -> float:
    """Return m = m* g(C / C*) without changing state or drawing randomness.

    Args:
        pressure: Current nonnegative weighted count C.
        reference_pressure: Positive reference count C* in the same units.
        reference_multiplier: Nonnegative calibrated newborn multiplier m*.
        response: Dimensionless density response; calibration requires g(1)=1.

    Returns:
        A finite nonnegative multiplier, which may exceed one.

    Raises:
        ValueError: If an input or the resulting multiplier is invalid.
    """
    values = (pressure, reference_pressure, reference_multiplier)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Density inputs must be finite")
    if pressure < 0 or reference_pressure <= 0 or reference_multiplier < 0:
        raise ValueError("Require C >= 0, C* > 0, and m* >= 0")
    ratio = pressure / reference_pressure
    if not np.isfinite(ratio):
        raise ValueError("Relative pressure must be finite")
    factor = float(response(ratio))
    multiplier = reference_multiplier * factor
    if not np.isfinite(factor) or factor < 0 or not np.isfinite(multiplier):
        raise ValueError("Response and multiplier must be finite and nonnegative")
    return multiplier


def beverton_holt(x: float, *, r: float = 2.0) -> float:
    """Evaluate the existing normalized curve r / (1 + (r - 1) x).

    Here r is g(0), not an independently established community growth rate.
    """
    if not np.isfinite(r) or r < 1 or not np.isfinite(x) or x < 0:
        raise ValueError("Beverton-Holt requires finite r >= 1 and x >= 0")
    return r / (1.0 + (r - 1.0) * x)


def _validate_counts(counts: Counts) -> None:
    if counts.shape != (2, 2) or not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Counts must be a finite nonnegative (sex=2, age=2) array")


@dataclass(frozen=True)
class Demography:
    """Baseline reproduction and survival for one demo population.

    Args:
        eggs_per_female: Expected newborns per reproductive female per step.
        juvenile_survival: Female and male age-0 baseline survival probabilities.
        female_fraction: Female fraction among newborns; both sexes must exist.
    """

    eggs_per_female: float
    juvenile_survival: tuple[float, float] = (0.5, 0.5)
    female_fraction: float = 0.5

    def __post_init__(self) -> None:
        values = (self.eggs_per_female, self.female_fraction, *self.juvenile_survival)
        if (
            not all(np.isfinite(value) for value in values)
            or self.eggs_per_female <= 0
            or not 0 < self.female_fraction < 1
            or len(self.juvenile_survival) != 2
            or any(not 0 < value <= 1 for value in self.juvenile_survival)
        ):
            raise ValueError("Require positive eggs, two positive survival probabilities, and 0 < female_fraction < 1")

    def reference_distribution(self, age1_total: float) -> Counts:
        """Derive a pre-reproduction reference with the surviving sex ratio.

        The target is the age-1 total, not a post-reproduction census total.
        """
        if not np.isfinite(age1_total) or age1_total <= 0:
            raise ValueError("Reference age-1 total must be finite and positive")
        mass = np.array([self.female_fraction, 1 - self.female_fraction])
        mass *= self.juvenile_survival
        result = np.zeros((2, 2))
        result[:, 1] = age1_total * mass / mass.sum()
        return result

    def reproduce(self, counts: Counts) -> Counts:
        """Return post-reproduction counts; inputs must be at the step boundary."""
        _validate_counts(counts)
        if np.any(counts[:, 0] != 0):
            raise ValueError("Pre-reproduction age 0 must be empty in this demo")
        result = counts.copy()
        eggs = float(counts[0, 1]) * self.eggs_per_female
        result[:, 0] = eggs * np.array([self.female_fraction, 1 - self.female_fraction])
        _validate_counts(result)
        return result

    def survive_and_age(self, counts: Counts) -> Counts:
        """Apply baseline survival once; replace old adults with surviving newborns."""
        _validate_counts(counts)
        result = np.zeros((2, 2))
        result[:, 1] = counts[:, 0] * self.juvenile_survival
        return result


@dataclass(frozen=True)
class WeightedPressure:
    """Sum one age across sexes and populations, using explicit nonnegative weights."""

    age: int
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.age not in (0, 1) or not self.weights:
            raise ValueError("Pressure needs age 0 or 1 and at least one participant")
        if any(not np.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Pressure weights must be finite and nonnegative")

    def __call__(self, community: Community) -> float:
        """Evaluate the same pressure definition on actual or reference counts."""
        return sum(
            weight * float(community[name][:, self.age].sum())
            for name, weight in self.weights.items()
        )


@dataclass(frozen=True)
class Calibration:
    """Derived values only; reference distributions remain the calibration inputs."""

    reference_pressure: float
    reference_multiplier: float


@dataclass(frozen=True)
class DensityHook:
    """Demo hook applying one pressure response to one population's newborns."""

    target: str
    pressure: Pressure
    response: Response = beverton_holt

    def calibrate(
        self,
        demography: Mapping[str, Demography],
        reference: Community,
    ) -> Calibration:
        """Derive C* and m* from the reference at the correct lifecycle boundary.

        References may be explicit or derived. This restricted two-age model
        rejects references whose sex composition cannot be maintained by a
        common newborn multiplier. It does not solve general equilibria.
        """
        if not np.isclose(self.response(1.0), 1.0, rtol=0, atol=1e-14):
            raise ValueError("A calibrated response must satisfy g(1) = 1")
        post_birth = {
            name: model.reproduce(reference[name])
            for name, model in demography.items()
        }
        target_model = demography[self.target]
        unregulated = target_model.survive_and_age(post_birth[self.target])
        produced_survivors = float(unregulated[:, 1].sum())
        target_adults = float(reference[self.target][:, 1].sum())
        if produced_survivors <= 0 or target_adults <= 0:
            raise ValueError("Calibration requires positive reference recruitment")
        multiplier = target_adults / produced_survivors
        # A total-only check would accept an inconsistent reference sex ratio.
        if not np.allclose(unregulated * multiplier, reference[self.target], rtol=1e-12, atol=0):
            raise ValueError("Reference sex composition is not an equilibrium of this demo")
        pressure = self.pressure(post_birth)
        density_multiplier(pressure, pressure, multiplier, response=self.response)
        return Calibration(pressure, multiplier)


def advance(
    demography: Mapping[str, Demography],
    counts: Community,
    reference: Community,
    hooks: Sequence[DensityHook],
) -> tuple[dict[str, Counts], dict[str, float]]:
    """Advance one deterministic step with all pressures read before any scaling.

    Recalibration is explicit here on every step for readability: it uses
    baseline demography and fixed reference inputs, never the actual census.
    One hook per population avoids an undefined composition of regulators.
    """
    targets = [hook.target for hook in hooks]
    if len(targets) != len(set(targets)) or set(targets) != set(demography):
        raise ValueError("Provide exactly one density hook per population")
    post_birth = {name: model.reproduce(counts[name]) for name, model in demography.items()}
    multipliers: dict[str, float] = {}
    for hook in hooks:
        calibration = hook.calibrate(demography, reference)
        multipliers[hook.target] = density_multiplier(
            hook.pressure(post_birth),
            calibration.reference_pressure,
            calibration.reference_multiplier,
            response=hook.response,
        )
    # Every hook observes the unregulated post-birth community, so reversing
    # the hook list cannot change another species' competition pressure.
    for name, multiplier in multipliers.items():
        post_birth[name][:, 0] *= multiplier
    result = {name: model.survive_and_age(post_birth[name]) for name, model in demography.items()}
    return result, multipliers


def main() -> None:
    """Show calibration, pressure choices, multispecies coupling, and recalibration."""
    model = Demography(eggs_per_female=10)
    demography = {"A": model}
    # A reference can be derived from a target age-1 census...
    reference = {"A": model.reference_distribution(age1_total=200)}
    # ...or supplied explicitly at the SAME pre-reproduction boundary.
    explicit_reference = {"A": np.array([[0.0, 100.0], [0.0, 100.0]])}
    juvenile = DensityHook("A", WeightedPressure(age=0, weights={"A": 1.0}))
    adult = DensityHook("A", WeightedPressure(age=1, weights={"A": 1.0}))
    print("DEMO ONLY: deterministic newborn regulation; no production API changes")
    for label, hook in (("juvenile pressure", juvenile), ("adult pressure", adult)):
        calibration = hook.calibrate(demography, explicit_reference)
        print(f"\n{label}: C*={calibration.reference_pressure:.1f}, m*={calibration.reference_multiplier:.3f}")
        for fraction in (0.5, 1.0, 2.0):
            counts = {"A": reference["A"] * fraction}
            result, factors = advance(demography, counts, reference, [hook])
            print(f"  adults={200 * fraction:6.1f} -> {result['A'][:, 1].sum():7.3f}; multiplier={factors['A']:.4f}")

    # The proportional demo has identical juvenile/adult pressure ratios.
    # A current-step birth pulse distinguishes the two pressure definitions.
    pulse = {"A": model.reproduce(reference["A"])}
    pulse["A"][:, 0] *= 2
    print("\nBirth pulse, fixed baseline/reference: newborns doubled, adults unchanged")
    for label, hook in (("juvenile", juvenile), ("adult", adult)):
        calibration = hook.calibrate(demography, reference)
        factor = density_multiplier(hook.pressure(pulse), calibration.reference_pressure, calibration.reference_multiplier, response=hook.response)
        print(f"  {label}: pressure={hook.pressure(pulse):.1f}, multiplier={factor:.4f}")

    # Each target has its own pressure definition and calibration. Shared
    # pressure does not imply a single universal multiplier for all species.
    community = {"A": model, "B": Demography(eggs_per_female=8, juvenile_survival=(0.4, 0.4))}
    reference = {name: item.reference_distribution(total) for (name, item), total in zip(community.items(), (200, 100), strict=True)}
    hooks = [
        DensityHook("A", WeightedPressure(1, {"A": 1.0, "B": 0.5})),
        DensityHook("B", WeightedPressure(1, {"A": 0.25, "B": 1.0}), partial(beverton_holt, r=3.0)),
    ]
    counts = {"A": reference["A"].copy(), "B": reference["B"] * 2}
    print("\nTwo populations: fixed reference A=200, B=100; start A=200, B=200")
    for tick in range(6):
        print(f"  tick={tick}: A={counts['A'][:, 1].sum():.3f}, B={counts['B'][:, 1].sum():.3f}")
        counts, _ = advance(community, counts, reference, hooks)

    # A baseline fecundity change is different from the temporary birth pulse.
    # This model's policy retains the target reference and recalibrates m*.
    changed = {"A": Demography(eggs_per_female=20)}
    old = adult.calibrate(demography, explicit_reference)
    new = adult.calibrate(changed, explicit_reference)
    result, _ = advance(changed, explicit_reference, explicit_reference, [adult])
    print(f"\nBaseline eggs 10 -> 20: m* {old.reference_multiplier:.3f} -> {new.reference_multiplier:.3f}; reference adults remain {result['A'][:, 1].sum():.1f}")


if __name__ == "__main__":
    main()
