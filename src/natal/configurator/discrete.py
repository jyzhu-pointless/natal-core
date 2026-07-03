"""Discrete-generation configurator (non-overlapping generations, Wright-Fisher).

Two age classes (age-0 juveniles, age-1 adults).  Non-overlapping
generations — adults are replaced each tick.

Create via ``Configurator.for_discrete(species)`` or
``DiscreteGenerationPopulation.setup(species)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from natal.configurator._base import Configurator, HookMap, set_param
from natal.data import (
    BEVERTON_HOLT,
    CONCAVE,
    FIXED,
    LINEAR,
    LOGISTIC,
    NO_COMPETITION,
)

if TYPE_CHECKING:
    from natal.population.discrete_generation import DiscreteGenerationPopulation

__all__ = ["DiscreteConfigurator"]


class DiscreteConfigurator(Configurator):
    """Configurator for ``DiscreteGenerationPopulation`` (discrete-generation model).

    Two age classes (age-0 juveniles, age-1 adults).  Non-overlapping
    generations — adults are replaced each tick.

    Create via ``Configurator.for_discrete(species)`` or
    ``DiscreteGenerationPopulation.setup(species)``.
    """

    def competition(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        age_1_carrying_capacity: float | None = None,
    ) -> DiscreteConfigurator:
        """Configure density-dependent competition.

        Args:
            carrying_capacity: Equilibrium total adults at age 1 (K).
            low_density_growth_rate: Per-capita growth at low density (r).
            juvenile_growth_mode: ``"concave"``, ``"logistic"``, … or int.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.

        Returns:
            Self for chaining.

        Note:
            When *carrying_capacity* (K) is set and the user has not
            explicitly called ``expected_num_adult_females=``,
            ``expected_num_adult_females`` is auto-computed as
            ``K * sex_ratio`` for the discrete model.
        """
        self._has_domain_params = True
        mode_value: int | None = None
        if isinstance(juvenile_growth_mode, str):
            _MODE_MAP: dict[str, int] = {
                "concave": CONCAVE, "linear": LINEAR, "logistic": LOGISTIC,
                "beverton_holt": BEVERTON_HOLT, "fixed": FIXED,
                "no_competition": NO_COMPETITION,
            }
            mode_value = _MODE_MAP.get(juvenile_growth_mode.lower())
            if mode_value is None:
                raise ValueError(
                    f"Unknown growth mode string: {juvenile_growth_mode!r}. "
                    f"Expected one of: {', '.join(sorted(_MODE_MAP))}."
                )
        elif juvenile_growth_mode is not None:
            mode_value = juvenile_growth_mode
        k_value = carrying_capacity
        if k_value is None and age_1_carrying_capacity is not None:
            k_value = age_1_carrying_capacity
        # K auto-detection: only during initial build.
        if k_value is None and self._pop_ref is None:
            init_ind = self._config.initial_individual_count
            if init_ind.size > 0 and init_ind.ndim >= 2 and init_ind.shape[1] >= 2:
                age_1_count = float(init_ind[:, 1, :].sum())
                if age_1_count >= 0.5:
                    k_value = age_1_count
                else:
                    total = float(init_ind.sum())
                    if total >= 0.5:
                        k_value = total
        for name, value in [
            ("carrying_capacity", k_value),
            ("low_density_growth_rate", low_density_growth_rate),
            ("juvenile_growth_mode", mode_value),
        ]:
            if value is not None:
                set_param(self._config, f"competition.{name}", value)
        if k_value is not None:
            self._sync_equilibrium()
        # Auto-compute expected_num_adult_females = K × sex_ratio
        if not getattr(self, "_has_user_expected_females", False):
            k = float(self._config.carrying_capacity)
            sr = float(self._config.sex_ratio)
            self._user_expected_adult_females = k * sr
        return self

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        female_adult_mating_rate: float | None = None,
        male_adult_mating_rate: float | None = None,
        fixed_egg_count: bool | None = None,
    ) -> DiscreteConfigurator:
        """Configure reproduction for the discrete-generation model.

        Args:
            eggs_per_female: Eggs per reproducing female per tick.
            sex_ratio: Female fraction of offspring (0–1).
            female_adult_mating_rate: Adult female mating probability.
            male_adult_mating_rate: Adult male mating probability.
            fixed_egg_count: Disable Poisson noise.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        # 0-d ndarray fields — write in-place, no staleness risk.
        if eggs_per_female is not None:
            set_param(self._config, "reproduction.eggs_per_female", eggs_per_female,
                      _sync_equilibrium=False)
        if sex_ratio is not None:
            set_param(self._config, "reproduction.sex_ratio", sex_ratio,
                      _sync_equilibrium=False)
        if eggs_per_female is not None or sex_ratio is not None:
            self._sync_equilibrium()
        # Scalars — write to config immediately (runtime) and store for build().
        scalar_overrides: dict[str, float] = {}
        if female_adult_mating_rate is not None:
            val = float(female_adult_mating_rate)
            self._female_adult_mating_rate = val
            scalar_overrides["female_adult_mating_rate"] = val
        if male_adult_mating_rate is not None:
            val = float(male_adult_mating_rate)
            self._male_adult_mating_rate = val
            scalar_overrides["male_adult_mating_rate"] = val
        if scalar_overrides:
            self._config = self._config._replace(**scalar_overrides)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        # Boolean flag — must use _replace (not a 0-d ndarray).
        if fixed_egg_count is not None:
            self._config = self._config._replace(fixed_egg_count=fixed_egg_count)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        return self

    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[str, float | Sequence[int | float] | Mapping[int, int | float]]],
        sperm_storage: Mapping[str, Mapping[str, float | Sequence[int | float] | Mapping[int, int | float]]] | None = None,
    ) -> DiscreteConfigurator:
        """Set the initial population for a discrete-generation model.

        Uses the discrete resolution (flat JSON-style dict) rather than the
        age-structured resolution.

        Args:
            individual_count: Per-sex, per-genotype initial counts.
                Nested as ``{sex: {genotype_selector: count}}``.
            sperm_storage: Ignored for discrete-generation models
                (sperm storage is not applicable).

        Returns:
            Self for chaining.
        """
        if not self._species:
            raise RuntimeError(
                "initial_state() requires a Species. "
                "Use Configurator.for_discrete() or setup(species, ...)."
            )
        from natal.configurator._factory import PopulationConfigBuilder

        array = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
            species=self._species,
            distribution=individual_count,
        )
        self._config = self._config._replace(
            initial_individual_count=array,
        )
        # Discrete models don't use sperm storage.
        if sperm_storage is not None:
            import warnings
            warnings.warn(
                "sperm_storage is ignored for discrete-generation populations.",
                UserWarning, stacklevel=2,
            )
        return self

    def survival(
        self,
        *,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
    ) -> DiscreteConfigurator:
        """Configure survival.  Only age-0 (juvenile→adult) matters.

        Both default to 1.0.  ``adult_survival`` is NOT accepted — in
        discrete-generation models, adults are fully replaced each tick,
        so adult survival is always 0.0 and cannot be overridden.

        Args:
            female_age0_survival: Female juvenile survival probability.
            male_age0_survival: Male juvenile survival probability.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        overrides: dict[str, float] = {}
        if female_age0_survival is not None:
            val = float(female_age0_survival)
            self._female_age0_survival = val
            overrides["female_age0_survival"] = val
        if male_age0_survival is not None:
            val = float(male_age0_survival)
            self._male_age0_survival = val
            overrides["male_age0_survival"] = val
        if overrides:
            self._config = self._config._replace(**overrides)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        return self

    def build(
        self, name: str | None = None, hooks: HookMap | None = None,
    ) -> DiscreteGenerationPopulation:  # type: ignore[name-defined]  # noqa: F821
        """Build and return a ``DiscreteGenerationPopulation``.

        Extracts discrete-specific scalars from stored override values
        before handing off to the base ``build()`` for finalisation.
        """
        # Extract scalars that replace the default values burned into
        # DiscretePopulationConfig at construction time.  This is the
        # only correct place — survival() and reproduction() store
        # overrides here, and the engine reads these scalars directly.
        overrides: dict[str, Any] = {}
        if self._female_adult_mating_rate is not None:
            overrides["female_adult_mating_rate"] = self._female_adult_mating_rate
        if self._male_adult_mating_rate is not None:
            overrides["male_adult_mating_rate"] = self._male_adult_mating_rate
        if self._female_age0_survival is not None:
            overrides["female_age0_survival"] = self._female_age0_survival
        if self._male_age0_survival is not None:
            overrides["male_age0_survival"] = self._male_age0_survival
        if overrides:
            self._config = self._config._replace(**overrides)

        from natal.population.discrete_generation import (
            DiscreteGenerationPopulation as DGP,
        )
        result = super().build(name=name, hooks=hooks)
        return cast(DGP, result)
