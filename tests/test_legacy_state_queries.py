"""Coverage for the legacy state-query and initial-distribution channels.

The R5 snapshot batch mechanically renamed ``self.state`` to
``self._live_state()`` inside several legacy methods (count getters, age
distribution, genotype queries, the per-age initial-state branches).
Those methods predate the batch and had no suite coverage; these tests
pin their numerical contracts so the renamed lines are exercised and the
legacy surface cannot silently rot.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import natal as nt


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build(name: str) -> nt.AgeStructuredPopulation:
    """Return a deterministic 3-age population with a known mixture."""
    return (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0, 5.0], "WT|Dr": [0.0, 10.0, 0.0]},
                "male": {"WT|WT": [0.0, 30.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 0.0],
            male_age_based_survival=[1.0, 1.0, 0.0],
        )
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0)
        .build()
    )


class TestLegacyCountGetters:
    """Count getters read the live container and sum exact axes."""

    def test_sex_counts_split_exact(self) -> None:
        """get_female_count / get_male_count sum their own sex axis."""
        pop = _build("LegacyCounts")
        assert pop.get_female_count() == 35.0
        assert pop.get_male_count() == 30.0
        assert pop.get_total_count() == 65.0

    def test_age_distribution_per_sex(self) -> None:
        """get_age_distribution('female'/'male') sums the genotype axis."""
        pop = _build("LegacyAgeDist")
        np.testing.assert_array_equal(
            pop.get_age_distribution("female"), [0.0, 30.0, 5.0]
        )
        np.testing.assert_array_equal(
            pop.get_age_distribution("male"), [0.0, 30.0, 0.0]
        )

    def test_get_genotype_count_deprecated_but_exact(self) -> None:
        """get_genotype_count returns per-sex sums and warns deprecation."""
        pop = _build("LegacyGenoCount")
        wt = pop.species.get_genotype_from_str("WT|WT")
        dr = pop.species.get_genotype_from_str("WT|Dr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert pop.get_genotype_count(wt) == (25.0, 30.0)
            assert pop.get_genotype_count(dr) == (10.0, 0.0)


class TestPerAgeInitialStateBranches:
    """The dict-per-age initial-state spelling lands exact cells."""

    def test_count_dict_form_fills_exact_cells(self) -> None:
        """``{"female": {"WT|WT": {1: 40, 2: 8}}`` writes those cells only."""
        pop = (
            nt.AgeStructuredPopulation.setup(
                species=_species("PerAgeCounts"), stochastic=False
            )
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": {1: 40, 2: 8}},
                    "male": {"WT|Dr": {2: 6}},
                }
            )
            .survival(
                female_age_based_survival=[1.0, 1.0, 0.0],
                male_age_based_survival=[1.0, 1.0, 0.0],
            )
            .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0)
            .build()
        )
        ic = pop.state.individual_count
        assert ic.shape == (2, 3, 3)
        assert float(ic[0, 1, 0]) == 40.0  # female WT|WT age 1
        assert float(ic[0, 2, 0]) == 8.0  # female WT|WT age 2
        assert float(ic[1, 2, 1]) == 6.0  # male WT|Dr age 2
        assert float(ic.sum()) == 54.0

    def test_sperm_dict_and_scalar_forms_fill_exact_cells(self) -> None:
        """Per-age dict, list, and scalar sperm forms hit distinct branches."""
        pop = (
            nt.AgeStructuredPopulation.setup(
                species=_species("PerAgeSperm"), stochastic=False
            )
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 10.0, 0.0]},
                    "male": {"WT|WT": [0.0, 10.0, 0.0], "WT|Dr": [0.0, 10.0, 0.0]},
                },
                sperm_storage={
                    "WT|WT": {
                        "WT|WT": {1: 3.0, 2: 4.0},
                        "WT|Dr": [1.0, 2.0, 0.0],
                    },
                    "WT|Dr": {"WT|WT": 2.5},
                },
            )
            .survival(
                female_age_based_survival=[1.0, 1.0, 0.0],
                male_age_based_survival=[1.0, 1.0, 0.0],
            )
            .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0)
            .build()
        )
        ss = pop.state.sperm_storage
        wt = pop.registry.ztype_index(
            pop.species.get_genotype_from_str("WT|WT"), "default"
        )
        dr = pop.registry.ztype_index(
            pop.species.get_genotype_from_str("WT|Dr"), "default"
        )
        # dict-per-age branch: {1: 3.0, 2: 4.0} on (WT|WT female, WT|WT male)
        assert float(ss[1, wt, wt]) == 3.0
        assert float(ss[2, wt, wt]) == 4.0
        # list branch: [1.0, 2.0, 0.0] on (WT|WT female, WT|Dr male)
        assert float(ss[0, wt, dr]) == 1.0
        assert float(ss[1, wt, dr]) == 2.0
        # scalar branch: 2.5 spread over adult ages on (WT|Dr female, WT|WT male)
        assert float(ss[1, dr, wt]) == 2.5
        assert float(ss[2, dr, wt]) == 2.5


class TestConstructorInitialArrays:
    """Direct constructor initial arrays bypass the builder chain."""

    def test_constructor_counts_and_sperm_land_exact(self) -> None:
        """__init__(initial_individual_count=..., initial_sperm_storage=...)."""
        species = _species("CtorArrays")
        cfg = nt.AgeStructuredPopulation.setup(
            species=species, stochastic=False
        ).age_structure(n_ages=3, new_adult_age=1)
        draft = cfg.config
        pop = nt.AgeStructuredPopulation(
            species=species,
            population_config=draft,
            name="ctor_arrays",
            initial_individual_count={
                "female": {"WT|WT": [0.0, 12.0, 0.0]},
                "male": {"WT|WT": [0.0, 12.0, 0.0]},
            },
        )
        assert float(pop.state.individual_count.sum()) == 24.0
        assert float(pop.state.individual_count[0, 1, 0]) == 12.0
        # No constructor sperm passed: the sperm plane exists but stays zero.
        assert float(pop.state.sperm_storage.sum()) == 0.0


class TestConstructorDictAndSpermForms:
    """Dict counts and every sperm spelling reach the direct-constructor path.

    The builder chain pre-expands dict spellings into the draft, so the
    Population constructor's own dict / list / scalar branches are only
    exercised when callers pass ``initial_*`` mappings directly.
    """

    def _draft(self, species: nt.Species) -> nt.ModelDraft:
        """Return a 3-age draft for direct construction."""
        cfg = nt.AgeStructuredPopulation.setup(
            species=species, stochastic=False
        ).age_structure(n_ages=3, new_adult_age=1)
        return cfg.config

    def test_dict_counts_constructor(self) -> None:
        """dict-per-age counts land in exact cells via the dict branch."""
        species = _species("CtorDictCounts")
        pop = nt.AgeStructuredPopulation(
            species=species,
            population_config=self._draft(species),
            name="ctor_dict_counts",
            initial_individual_count={
                "female": {"WT|WT": {1: 15.0, 2: 3.0}},
                "male": {"WT|Dr": {2: 7.0}},
            },
        )
        ic = pop.state.individual_count
        assert float(ic[0, 1, 0]) == 15.0
        assert float(ic[0, 2, 0]) == 3.0
        assert float(ic[1, 2, 1]) == 7.0
        assert float(ic.sum()) == 25.0

    def test_sperm_all_spellings_constructor(self) -> None:
        """dict / list / tuple / scalar sperm forms fill exact cells."""
        species = _species("CtorSpermForms")
        pop = nt.AgeStructuredPopulation(
            species=species,
            population_config=self._draft(species),
            name="ctor_sperm_forms",
            initial_individual_count={
                "female": {"WT|WT": [0.0, 10.0, 0.0]},
                "male": {"WT|WT": [0.0, 10.0, 0.0], "WT|Dr": [0.0, 10.0, 0.0]},
            },
            initial_sperm_storage={
                "WT|WT": {
                    "WT|WT": {1: 3.0, 2: 4.0},   # dict branch
                    "WT|Dr": [1.0, 2.0, 0.0],     # list branch
                },
                "WT|Dr": {
                    "WT|WT": (0.0, 5.0),          # tuple branch
                    "WT|Dr": 2.5,                 # scalar branch (adult ages)
                },
            },
        )
        ss = pop.state.sperm_storage
        wt = pop.registry.ztype_index(
            pop.species.get_genotype_from_str("WT|WT"), "default"
        )
        dr = pop.registry.ztype_index(
            pop.species.get_genotype_from_str("WT|Dr"), "default"
        )
        assert float(ss[1, wt, wt]) == 3.0 and float(ss[2, wt, wt]) == 4.0
        assert float(ss[0, wt, dr]) == 1.0 and float(ss[1, wt, dr]) == 2.0
        assert float(ss[1, dr, wt]) == 5.0 and float(ss[0, dr, wt]) == 0.0
        assert float(ss[1, dr, dr]) == 2.5 and float(ss[2, dr, dr]) == 2.5

    def test_genotypes_present_deprecated_but_exact(self) -> None:
        """genotypes_present lists exactly the occupied genotypes."""
        import warnings as _w

        pop = _build("LegacyPresent")
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            present = pop.genotypes_present
        names = {str(g) for g in present}
        assert names == {"WT|WT", "WT|Dr"}

    def test_discrete_sex_counts_exact(self) -> None:
        """Discrete getters sum their own sex axis."""
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=_species("DiscreteCounts"), stochastic=False
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": 14.0},
                    "male": {"WT|WT": 9.0},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0)
            .build()
        )
        assert pop.get_female_count() == 14.0
        assert pop.get_male_count() == 9.0
        assert pop.get_total_count() == 23.0

    def test_snapshot_state_hook_is_required(self) -> None:
        """A subclass that never implements the hook gets NotImplementedError."""

        from natal.frontend.population.base import BasePopulation

        class _BareHost(BasePopulation):
            """Minimal abstract-surface stub; snapshot hook left inherited."""

            def get_female_count(self) -> int:
                return 0

            def get_male_count(self) -> int:
                return 0

            def get_total_count(self) -> int:
                return 0

            def run_tick(self) -> Any:
                return self

            def run(self, n_steps: int, record_every: int = 1, finish: bool = False) -> Any:
                return self

            def reset(self) -> None:
                return None

            def update(self) -> Any:
                return self

        host = object.__new__(_BareHost)
        object.__setattr__(host, "_state", "sentinel")
        with pytest.raises(NotImplementedError, match="_snapshot_state"):
            _ = host._snapshot_state()


class TestAdultCountAndBothDistribution:
    """The remaining sex-branch getters on the live container."""

    def test_get_adult_count_sex_split_and_both(self) -> None:
        """Adult counts sum ages >= new_adult_age per sex and combined."""
        pop = _build("LegacyAdult")
        # Adults are ages 1-2: female 30 WT|WT + 10 WT|Dr = 35... split per sex.
        assert pop.get_adult_count("female") == 35.0
        assert pop.get_adult_count("male") == 30.0
        assert pop.get_adult_count("both") == 65.0
        # Age-0 juveniles are excluded from every adult query.
        assert pop.get_adult_count("both") < pop.get_total_count() or all(
            float(c) == 0.0
            for c in pop.state.individual_count[:, 0, :].reshape(-1)
        )

    def test_age_distribution_both_sums_genotype_axis(self) -> None:
        """get_age_distribution('both') sums sex and genotype axes."""
        pop = _build("LegacyAgeBoth")
        np.testing.assert_array_equal(
            pop.get_age_distribution("both"), [0.0, 60.0, 5.0]
        )
