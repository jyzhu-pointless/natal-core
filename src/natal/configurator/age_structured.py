"""Age-structured configurator (overlapping generations).

Supports arbitrary age classes with per-age survival, mating, and
fertility.  Adults survive across ticks — generations overlap.

The ``AgeStructuredConfigurator`` class provides chainable domain
methods that mutate config arrays in-place:

    cfg = AgeStructuredConfigurator.from_species(species)
    cfg.age_structure(8, 2).competition(K=10000)
    pop = cfg.build()

Per-age parameters accept flexible input: scalar, list, dict,
or callable.  They are resolved by ``resolve_age_param()``.

Create via ``Configurator.from_species()`` or
``AgeStructuredPopulation.setup(species, legacy_path=False)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from numpy.typing import NDArray

from natal.configurator._base import Configurator, HookMap, set_param
from natal.configurator._params import resolve_age_param
from natal.data import (
    BEVERTON_HOLT,
    CONCAVE,
    FIXED,
    LINEAR,
    LOGISTIC,
    NO_COMPETITION,
    DiscretePopulationConfig,
)

if TYPE_CHECKING:
    from natal.population.age_structured import AgeStructuredPopulation

__all__ = ["AgeStructuredConfigurator"]


class AgeStructuredConfigurator(Configurator):
    """Configurator for ``AgeStructuredPopulation`` (overlapping generations).

    Supports arbitrary age classes with per-age survival, mating, and
    fertility.  Adults survive across ticks — generations overlap.

    Per-age parameters accept flexible input: scalar, list, dict, callable.

    Create via ``Configurator.from_species()`` or
    ``AgeStructuredPopulation.setup(species, legacy_path=False)``.
    """

    def age_structure(
        self, n_ages: int, new_adult_age: int,
        generation_time: float | None = None,
    ) -> AgeStructuredConfigurator:
        """Lock population dimensions.

        Args:
            n_ages: Total number of age classes.
            new_adult_age: First adult age.
            generation_time: Optional marker for model interpretation.

        Returns:
            Self for chaining.

        Note:
            Must be called before any domain method (competition,
            reproduction, survival, etc.).  Calling it after domain
            methods will raise ``RuntimeError``.
        """
        if getattr(self, "_has_domain_params", False):
            raise RuntimeError(
                "age_structure() must be called before any domain method "
                "(competition(), reproduction(), survival(), etc.). "
                "Domain methods have already been called on this configurator."
            )
        if n_ages <= 1:
            raise ValueError(f"n_ages must be at least 2, got {n_ages}")
        if new_adult_age < 0 or new_adult_age >= n_ages:
            raise ValueError(
                f"new_adult_age must be in [0, {n_ages}), got {new_adult_age}"
            )
        from natal.data import build_population_config

        old = self._config
        # Use species blueprint maps (unexpanded) so that
        # build_population_config applies slab expansion exactly once.
        if self._species is not None:
            bp = self._species.get_config_blueprint()
            n_g_orig = bp["n_genotypes"]
            n_hg_orig = bp["n_gtypes"]
            z2g_bp = bp["zygotes_to_gametes_map"]
            g2z_bp = bp["gametes_to_zygotes_map"]
        else:
            n_g_orig = old.n_ztypes
            n_hg_orig = old.n_gtypes
            z2g_bp = old.zygotes_to_gametes_map
            g2z_bp = old.gametes_to_zygotes_map

        self._config = build_population_config(
            n_genotypes=n_g_orig,
            n_gtypes=n_hg_orig,
            n_ages=n_ages,
            n_glabs=old.n_glabs,
            n_slabs=old.n_slabs,
            gamete_labels=self._species.gamete_labels if self._species else None,
            somatic_labels=self._species.somatic_labels if self._species else None,
            zygotes_to_gametes_map=z2g_bp,
            gametes_to_zygotes_map=g2z_bp,
            new_adult_age=new_adult_age,
            generation_time=generation_time,
            stochastic=bool(old.stochastic),
            continuous_sampling=bool(old.continuous_sampling),
            fixed_egg_count=bool(old.fixed_egg_count),
            has_sex_chromosomes=old.has_sex_chromosomes,
        )
        # Rebuild registry for the new n_ages (affects genotype lookup dims).
        if self._species is not None:
            from natal.configurator._base import build_registry
            self._registry = build_registry(self._species)
        return self

    def competition(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        competition_strength: float | None = None,
        expected_num_adult_females: float | None = None,
        equilibrium_distribution: NDArray[np.float64] | None = None,
        age_1_carrying_capacity: float | None = None,
        old_juvenile_carrying_capacity: float | None = None,
    ) -> AgeStructuredConfigurator:
        """Configure density-dependent competition.

        Args:
            carrying_capacity: Equilibrium population at age 1 (K).
            low_density_growth_rate: Per-capita growth at low density (r).
            juvenile_growth_mode: Regulation function (string or int).
            competition_strength: Larval competition weight.
            expected_num_adult_females: Target adult females (Champer model).
            equilibrium_distribution: Custom (n_sexes, n_ages) array for
                Champer equilibrium computation.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
            old_juvenile_carrying_capacity: Legacy alias.

        Returns:
            Self for chaining.
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
        # ---- carrying capacity (K) fallback chain ----
        k_value = carrying_capacity
        if k_value is None and age_1_carrying_capacity is not None:
            k_value = age_1_carrying_capacity
        if k_value is None and old_juvenile_carrying_capacity is not None:
            k_value = old_juvenile_carrying_capacity
        # Only auto-detect K during initial build (no live Population).
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
            ("competition_strength", competition_strength),
        ]:
            if value is not None:
                set_param(self._config, f"competition.{name}", value)
        if expected_num_adult_females is not None:
            self._user_expected_adult_females = float(expected_num_adult_females)
            self._has_user_expected_females = True
        if equilibrium_distribution is not None:
            self._equilibrium_distribution = equilibrium_distribution
        if k_value is not None:
            self._sync_equilibrium()
        return self

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        sperm_displacement_rate: float | None = None,
        female_age_based_mating_rate: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male_age_based_mating_rate: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        age_based_reproduction_rate: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age_based_fertility: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        fixed_egg_count: bool | None = None,
    ) -> AgeStructuredConfigurator:
        """Configure reproduction for the age-structured model.

        Args:
            eggs_per_female: Base eggs per reproducing female.
            sex_ratio: Female fraction of offspring (0–1).
            sperm_displacement_rate: Fraction of stored sperm displaced.
            female_age_based_mating_rate: Per-age female mating probability.
            male_age_based_mating_rate: Per-age male mating probability.
            age_based_reproduction_rate: Per-age reproduction participation.
            female_age_based_fertility: Per-age fertility weight.
            fixed_egg_count: Disable Poisson noise.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        assert not isinstance(self._config, DiscretePopulationConfig), \
            "AgeStructuredConfigurator requires PopulationConfig"

        n_ages = self._config.n_ages
        for name, value in [
            ("eggs_per_female", eggs_per_female),
            ("sex_ratio", sex_ratio),
            ("sperm_displacement_rate", sperm_displacement_rate),
        ]:
            if value is not None:
                set_param(self._config, f"reproduction.{name}", value,
                          _sync_equilibrium=False)
        if eggs_per_female is not None or sex_ratio is not None:
            from natal.engine.simulation.age_structured import sync_equilibrium_metrics
            sync_equilibrium_metrics(self._config)
        if female_age_based_mating_rate is not None:
            self._config.age_based_mating_rates[0, :] = resolve_age_param(
                female_age_based_mating_rate, n_ages, np.zeros(n_ages))
        if male_age_based_mating_rate is not None:
            self._config.age_based_mating_rates[1, :] = resolve_age_param(
                male_age_based_mating_rate, n_ages, np.zeros(n_ages))
        if age_based_reproduction_rate is not None:
            self._config.age_based_reproduction_rates[:] = resolve_age_param(
                age_based_reproduction_rate, n_ages, np.ones(n_ages))
        if female_age_based_fertility is not None:
            self._config.female_age_based_fertility[:] = resolve_age_param(
                female_age_based_fertility, n_ages, np.ones(n_ages))
        if fixed_egg_count is not None:
            self._config = self._config._replace(fixed_egg_count=fixed_egg_count)
        return self

    def survival(
        self,
        *,
        female_age_based_survival: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male_age_based_survival: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
    ) -> AgeStructuredConfigurator:
        """Configure survival rates. Per-age params accept flexible forms.

        Args:
            female_age_based_survival: Female survival rates (scalar, list, dict, or callable).
            male_age_based_survival: Male survival rates (same forms).

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        assert not isinstance(self._config, DiscretePopulationConfig), \
            "AgeStructuredConfigurator requires PopulationConfig"

        n_ages = self._config.n_ages
        if female_age_based_survival is not None:
            self._config.age_based_survival_rates[0, :] = (
                resolve_age_param(female_age_based_survival, n_ages, np.ones(n_ages)))
        if male_age_based_survival is not None:
            self._config.age_based_survival_rates[1, :] = (
                resolve_age_param(male_age_based_survival, n_ages, np.ones(n_ages)))
        return self

    def build(
        self, name: str | None = None, hooks: HookMap | None = None,
    ) -> AgeStructuredPopulation:  # type: ignore[name-defined]  # noqa: F821  # lazy-imported class forward ref
        """Finalise the config and create an ``AgeStructuredPopulation``.

        Delegates to :meth:`Configurator.build` which syncs equilibrium
        metrics, runs optional index compression, builds the custom
        array, merges hooks, and passes the config to the Population
        constructor.

        Args:
            name: Population name (falls back to ``.setup(name=...)``
                or ``"AgeStructuredPopulation"``).
            hooks: Additional hook registrations merged with any stored
                via :meth:`hooks`.

        Returns:
            An ``AgeStructuredPopulation`` ready for simulation.
        """
        from natal.population.age_structured import (
            AgeStructuredPopulation as ASP,
        )
        result = super().build(name=name, hooks=hooks)
        return cast(ASP, result)
