"""Params contract: every runtime-mutable value in one flat container.

``Params`` is a mutable object owning two kinds of content:

- **Ecology section** — values not indexed by genotype: five bounded
  scalars, the growth-mode selector, per-sex/per-age demographic
  vectors, the equilibrium declaration, and user-registered custom
  slots.  History snapshots and memory checkpoints copy this section.
- **Genetics section** — genotype-indexed tables that scale with
  z/g: the four fitness tensors, the offspring probability tensor, the
  meiosis map, and the zygote compatibility weights.  In spatial models
  this section moves into the variant bank (slice ⑤).

Section membership is data, not code structure: the route table
(slice ③) tags every parameter with its section and thereby its write
channel — scalars/vectors through :meth:`~natal.contracts.params.Params.apply`
style writers, genetics tables through per-tensor writes.

Ownership discipline: ``Params`` owns its arrays after materialization.
Writers mutate array *contents* in place (after validating shape);
nobody replaces an array object on a live population.  Scalars are
plain Python floats/ints — writers assign attributes, readers read
them; the ``config.field[()] = value`` 0-d idiom is gone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["CustomValue", "EcologySnapshot", "Params"]

CustomValue: TypeAlias = bool | int | float | NDArray[np.float64]
# One snapshot level deeper: the ecology mapping carries the custom-slots
# mapping as a nested value.
EcologySnapshot: TypeAlias = dict[str, "bool | int | float | NDArray[np.float64] | dict[str, CustomValue]"]


@dataclass(slots=True, eq=False)
class Params:
    """Mutable runtime parameters (see module docstring for sections).

    Attributes:
        carrying_capacity: Environment capacity K (individuals).
        eggs_per_female: Expected eggs per female per tick.
        sex_ratio: Newborn fraction female, in [0, 1].
        sperm_displacement_rate: Probability a new mating displaces
            stored sperm, in [0, 1].
        low_density_growth_rate: Intrinsic growth rate r at low
            density; feeds the density curve.
        growth_mode: Density-regulation selector — 0 no regulation,
            1 fixed cap, 2 linear, 3 Beverton-Holt (curve-name to id
            resolution happens at build; 4+ reserved for custom curve
            slots).
        external_expected_eggs: Champer-model egg override for the
            survival-rate calibration; negative means unused.
        survival_rates: (2, A) per-sex survival probabilities.
        mating_rates: (2, A) per-sex mating probabilities.
        reproduction_rates: (A,) female reproduction participation.
        fertility: (A,) relative female fertility per age.
        competition_weights: (A,) relative competition strength per
            age.
        equilibrium_distribution: (2, A) declared equilibrium age
            distribution; shape (0, 0) selects derivation mode (derive
            from current params when needed).
        migration_rate: (n_demes, n_sexes, A) migration-rate columns.
            Spatial models carry the per-deme/per-sex/per-age migration
            probability; runtime migration multiplies this rate column
            with the frozen Blueprint CSR.  Panmictic models keep one
            all-zero (1, n_sexes, A) column.  The shape-(0, 0, 0)
            default is the "not declared" sentinel used by bare
            constructions.
        custom_slots: User-registered named values (bool/int/float or
            a (sex, age, genotype) float array).
        viability_fitness: (2, A, z) viability coefficients.
        fecundity_fitness: (2, z) fecundity coefficients.
        sexual_selection_fitness: (z, z) female-x-male mating weights.
        zygote_viability_fitness: (2, z) zygote survival coefficients.
        offspring_tensor: (z, z, z) offspring ztype probabilities given
            (mother, father) ztypes.
        meiosis_map: (2, z, g) diploid-to-gamete probabilities.
        female_ztype_compatibility: (z,) female-side mating weights.
        male_ztype_compatibility: (z,) male-side mating weights.
    """

    # -- ecology: scalars ---------------------------------------------------
    carrying_capacity: float
    eggs_per_female: float
    sex_ratio: float
    sperm_displacement_rate: float
    low_density_growth_rate: float
    growth_mode: int
    external_expected_eggs: float
    # -- ecology: vectors ---------------------------------------------------
    survival_rates: NDArray[np.float64]
    mating_rates: NDArray[np.float64]
    reproduction_rates: NDArray[np.float64]
    fertility: NDArray[np.float64]
    competition_weights: NDArray[np.float64]
    equilibrium_distribution: NDArray[np.float64]
    # -- genetics: genotype-indexed tables ----------------------------------
    viability_fitness: NDArray[np.float64]
    fecundity_fitness: NDArray[np.float64]
    sexual_selection_fitness: NDArray[np.float64]
    zygote_viability_fitness: NDArray[np.float64]
    offspring_tensor: NDArray[np.float64]
    meiosis_map: NDArray[np.float64]
    female_ztype_compatibility: NDArray[np.float64]
    male_ztype_compatibility: NDArray[np.float64]
    # Spatial ecology column: (n_demes, n_sexes, n_ages).  Dataclass rule
    # forces the default after every non-default field and before
    # ``custom_slots``; (0, 0, 0) means "not declared" — materialization
    # always fills the real shape.
    migration_rate: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float64)
    )
    # defaulted field must come last (dataclass rule)
    custom_slots: dict[str, CustomValue] = field(default_factory=dict[str, CustomValue])

    def snapshot_ecology(self) -> EcologySnapshot:
        """Copy the ecology section into a plain mapping.

        Used by history snapshots and memory checkpoints (the genetics
        section is deliberately excluded — it is a "mod", not a save).

        Returns:
            A fresh mapping of every ecology field name to its value;
            arrays and the custom-slot dict are shallow-copied at the
            top level (array contents are not duplicated — callers that
            need isolation copy further).
        """
        snapshot: EcologySnapshot = {
            "carrying_capacity": self.carrying_capacity,
            "eggs_per_female": self.eggs_per_female,
            "sex_ratio": self.sex_ratio,
            "sperm_displacement_rate": self.sperm_displacement_rate,
            "low_density_growth_rate": self.low_density_growth_rate,
            "growth_mode": self.growth_mode,
            "external_expected_eggs": self.external_expected_eggs,
            "survival_rates": self.survival_rates,
            "mating_rates": self.mating_rates,
            "reproduction_rates": self.reproduction_rates,
            "fertility": self.fertility,
            "competition_weights": self.competition_weights,
            "equilibrium_distribution": self.equilibrium_distribution,
            "migration_rate": self.migration_rate,
        }
        # Nested mapping is its own level; store a shallow copy of the slots.
        snapshot["custom_slots"] = dict[str, CustomValue](self.custom_slots)
        return snapshot
