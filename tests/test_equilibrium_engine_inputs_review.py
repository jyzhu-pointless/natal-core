"""Evaluator-added coverage for the engine-read calibration fixes (F1/F3).

Companion to ``tests/test_equilibrium_engine_inputs.py``.  That module pins
the plain carrying-capacity path in both engines; this one closes the gaps the
review identified in the paths the same two rules also reach, or should reach:

- the **declared** ``equilibrium_distribution`` branch, whose age-1 total and
  ``C*`` come from the declaration but whose ``s_0_avg`` still carries the
  offspring sex rule (a sex-chromosome declaration can therefore only be
  stationary at the *genetic* composition);
- ``external_expected_eggs`` (the Champer ``expected_num_new_adult_females``
  anchor), which is a calibration input produced *outside*
  ``equilibrium_metrics_core`` and must still be built from the weights the
  owning tick reads;
- the spatial demes handoff (``blueprint.discrete_generation`` /
  ``has_sex_chromosomes`` are read from the frozen spatial contract, not from
  the per-deme drafts);
- the fused Wright-Fisher tick (``extreme_speed_mode=3``), which owns its own
  copy of both rules.

Every assertion states the property it guards; the calibration pairs are
compared as raw ``float`` values (exact equality), because the contract is that
one rule has one implementation and both entries return its result unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt

K = 2000.0
S_F0, S_M0 = 0.9, 0.5
_XY_FEMALE = "A|A;X1|X1"
_XY_MALE = "A|A;X1|Y1"
#: Age-1 female share the sex-chromosome tick actually reaches (balanced
#: genetic offspring split filtered by each sex's own age-0 survival).
_GENETIC_FEMALE_SHARE = (0.5 * S_F0) / (0.5 * S_F0 + 0.5 * S_M0)


def _autosomal_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["W"]}}, gamete_labels=["default"]
    )


def _xy_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )


def _total(population: object) -> float:
    return float(np.asarray(population.state.individual_count).sum())


def _calibration(population: object) -> tuple[float, float]:
    """The ``pop.params`` query pair as the exact values the kernel returned."""
    return (
        float(population.params.expected_competition_strength),
        float(population.params.expected_survival_rate),
    )


def _xy_declared(
    name: str, *, sex_ratio: float, female_share: float
) -> tuple[nt.DiscreteGenerationPopulation, np.ndarray]:
    """XY discrete population declaring *female_share* of the age-1 total."""
    declared = np.array(
        [[0.0, K * female_share], [0.0, K * (1.0 - female_share)]], dtype=np.float64
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_xy_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {_XY_FEMALE: float(declared[0, 1])},
                "male": {_XY_MALE: float(declared[1, 1])},
            }
        )
        .survival(female_age0_survival=S_F0, male_age0_survival=S_M0)
        .reproduction(eggs_per_female=10.0, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
            equilibrium_distribution=declared,
        )
        .build()
    )
    return population, declared


def _xy_champer(name: str, *, sex_ratio: float) -> nt.DiscreteGenerationPopulation:
    """XY discrete population anchored by ``expected_num_new_adult_females``."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_xy_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={"female": {_XY_FEMALE: K / 2}, "male": {_XY_MALE: K / 2}}
        )
        .survival(female_age0_survival=S_F0, male_age0_survival=S_M0)
        .reproduction(eggs_per_female=10.0, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .competition(expected_num_new_adult_females=K / 2)
        .build()
    )


@pytest.mark.parametrize("sex_ratio", [0.2, 0.3, 0.5, 0.8])
def test_declared_equilibrium_ignores_sex_ratio_on_sex_chromosomes(
    sex_ratio: float,
) -> None:
    """The declared branch reaches ``s_0_avg`` through the same sex rule.

    ``C*`` is fixed by the declared composition, but ``s*`` blends the two
    age-0 survival rates by the offspring female fraction.  For a
    sex-chromosome species that fraction is the genetic split, so every stored
    ``sex_ratio`` must return one identical pair; before the fix ``s*`` (and
    through it the simulated equilibrium) followed the ignored parameter.
    """
    population, _ = _xy_declared(
        f"declared_{sex_ratio}", sex_ratio=sex_ratio, female_share=_GENETIC_FEMALE_SHARE
    )
    reference, _ = _xy_declared(
        "declared_reference", sex_ratio=0.5, female_share=_GENETIC_FEMALE_SHARE
    )
    assert _calibration(population) == _calibration(reference)


def test_declared_genetic_composition_is_the_stationary_state() -> None:
    """A declaration split at 1:1 cannot be stationary; the genetic one is.

    The engine's realised age-1 composition is the *genetic* offspring split
    filtered by age-0 survival.  Declaring exactly that composition must hold
    per sex to the convergence floor, so the calibrated ``K`` is the state the
    model reaches rather than one it drifts away from (the pre-fix rule left
    this declaration drifting to 2387.1).
    """
    population, declared = _xy_declared(
        "declared_stationary", sex_ratio=0.3, female_share=_GENETIC_FEMALE_SHARE
    )
    population.run(400)
    counts = np.asarray(population.state.individual_count)
    assert _total(population) == pytest.approx(float(declared.sum()), rel=1e-9)
    assert float(counts[0].sum()) == pytest.approx(float(declared[0, 1]), rel=1e-9)
    assert float(counts[1].sum()) == pytest.approx(float(declared[1, 1]), rel=1e-9)


@pytest.mark.parametrize("sex_ratio", [0.2, 0.3, 0.8])
def test_external_expected_eggs_keeps_the_genetic_sex_rule(sex_ratio: float) -> None:
    """The Champer anchor is a calibration input and shares the sex rule.

    ``expected_num_new_adult_females`` derives ``external_expected_eggs`` and
    leaves ``C*`` on the derived distribution, so both halves of the pair are
    built from the reference composition.  A sex-chromosome species must
    therefore report one pair for every stored ``sex_ratio``; both values moved
    before the fix (the derived split *and* ``s_0_avg``).
    """
    population = _xy_champer(f"champer_{sex_ratio}", sex_ratio=sex_ratio)
    reference = _xy_champer("champer_reference", sex_ratio=0.5)
    assert float(np.asarray(population.params.external_expected_eggs)) == float(
        np.asarray(reference.params.external_expected_eggs)
    )
    assert _calibration(population) == _calibration(reference)


def test_discrete_spatial_demes_ignore_a_fertility_write() -> None:
    """Spatial demes hand the flag through the frozen contract, not per draft.

    ``SpatialPopulation.blueprint`` materialises from the reference deme, and
    that is what the native session reads for ``discrete_generation``.  A raw
    fertility write to every deme must stay inert there; before the fix the
    spatial total moved from 4041 to 2004 (the written factor of 2).
    """
    population = (
        nt.SpatialPopulation.builder(
            _autosomal_species("spatial_discrete"), 2, pop_type="discrete_generation"
        )
        .initial_state(
            individual_count={"female": {"W|W": K / 2}, "male": {"W|W": K / 2}}
        )
        .survival(female_age0_survival=S_F0, male_age0_survival=S_M0)
        .reproduction(eggs_per_female=10.0, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .build()
    )
    assert population.blueprint.discrete_generation is True
    for deme in range(2):
        population.deme(deme).write_ecology("fertility", np.array([0.0, 2.0]))
    population.run(300)
    total = float(
        sum(
            float(np.asarray(population.deme(deme).state.individual_count).sum())
            for deme in range(2)
        )
    )
    # Two independent demes: the per-deme anchor is K.
    assert total == pytest.approx(2.0 * K, rel=0.03)


@pytest.mark.parametrize("fertility,moved", [(0.5, True), (2.0, False)])
def test_spatial_age_demes_read_fertility_as_the_tick_does(
    fertility: float, moved: bool
) -> None:
    """The spatial age path must resolve ``discrete_generation = False``.

    An in-domain weight is effective in both the calibration and the tick, so
    the pair must move; an out-of-domain weight clamps to 1 in both, so it must
    not.  A missing flag (or the discrete default) collapses both cases onto the
    inert reading and this test fails.
    """
    from natal.frontend.model.ecology import derive_equilibrium_metrics_from_draft

    population = (
        nt.SpatialPopulation.builder(
            _autosomal_species(f"spatial_age_{fertility}"), 2, pop_type="age_structured"
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"W|W": {1: K / 2}},
                "male": {"W|W": {1: K / 2}},
            }
        )
        .survival(
            female_age_based_survival=[1.0, S_F0],
            male_age_based_survival=[1.0, S_M0],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0],
            eggs_per_female=10.0,
            sex_ratio=0.5,
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .build()
    )
    assert population.blueprint.discrete_generation is False
    before = derive_equilibrium_metrics_from_draft(population.deme(0).config)
    for deme in range(2):
        population.deme(deme).write_ecology(
            "fertility", np.array([0.0, fertility], dtype=np.float64)
        )
    after = derive_equilibrium_metrics_from_draft(population.deme(0).config)
    assert (after != before) is moved


def test_sex_chromosome_spatial_demes_share_the_genetic_split() -> None:
    """Spatial XY demes must ignore ``sex_ratio`` like the single-deme engine.

    The spatial contract carries ``has_sex_chromosomes``; losing it would split
    the spatial reference composition by the stored ratio, which moved the
    calibration pair before the fix (8709.7/0.370 vs 12857.1/0.222).
    """
    from natal.frontend.model.ecology import derive_equilibrium_metrics_from_draft

    def build(name: str, sex_ratio: float) -> object:
        return (
            nt.SpatialPopulation.builder(
                _xy_species(name), 2, pop_type="discrete_generation"
            )
            .initial_state(
                individual_count={
                    "female": {_XY_FEMALE: K / 2},
                    "male": {_XY_MALE: K / 2},
                }
            )
            .survival(female_age0_survival=S_F0, male_age0_survival=S_M0)
            .reproduction(eggs_per_female=10.0, sex_ratio=sex_ratio)
            .competition(
                juvenile_growth_mode="beverton_holt",
                carrying_capacity=K,
                low_density_growth_rate=3.0,
            )
            .build()
        )

    biased = build("spatial_xy_biased", 0.3)
    balanced = build("spatial_xy_balanced", 0.5)
    assert biased.blueprint.has_sex_chromosomes is True
    assert derive_equilibrium_metrics_from_draft(
        biased.deme(0).config
    ) == derive_equilibrium_metrics_from_draft(balanced.deme(0).config)


@pytest.mark.parametrize("sex_ratio", [0.2, 0.3, 0.8])
def test_hook_metrics_context_uses_the_genetic_split(sex_ratio: float) -> None:
    """``TickContext.metrics`` rides the flat entry and needs both flags.

    Hook callbacks reach the same kernel through
    ``tick_context._equilibrium_metrics``, whose distribution comes from the
    live state.  Its ``s*`` blends the age-0 rates by the offspring sex
    fraction, so a sex-chromosome model must report one value for every stored
    ``sex_ratio``; before the fix the first-event ``s*`` was 0.3226 at 0.3
    against 0.2857 at 0.5.
    """
    captured: list[float] = []

    @nt.hook(event="first")
    def capture(ctx: nt.TickContext) -> int:
        captured.append(float(ctx.metrics.s_star))
        return 0

    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_xy_species(f"hook_{sex_ratio}"),
            name=f"hook_{sex_ratio}",
            stochastic=False,
        )
        .initial_state(
            individual_count={"female": {_XY_FEMALE: K / 2}, "male": {_XY_MALE: K / 2}}
        )
        .survival(female_age0_survival=S_F0, male_age0_survival=S_M0)
        .reproduction(eggs_per_female=10.0, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .hooks(capture)
        .build()
    )
    population.run(1)
    assert captured, "the first-event hook did not run"
    assert captured[0] == pytest.approx(2000.0 / (10000.0 * (0.5 * S_F0 + 0.5 * S_M0)))


def test_fused_wright_fisher_tick_ignores_a_fertility_write() -> None:
    """The fused tick owns its own fertility rule and reads none either.

    ``extreme_speed_mode=3`` bypasses the staged lifecycle, so it is a separate
    owner of the discrete rule; a raw fertility write must leave both the
    calibration and the realised total at ``K`` (before the fix the total was
    1000).
    """
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_autosomal_species("wf_fused"),
            name="wf_fused",
            stochastic=False,
            extreme_speed_mode=3,
        )
        .initial_state(
            individual_count={"female": {"W|W": K / 2}, "male": {"W|W": K / 2}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .build()
    )
    before = _calibration(population)
    population.params.tensor_write("fertility", np.array([0.0, 2.0]))
    assert _calibration(population) == before
    population.run(400)
    assert _total(population) == pytest.approx(K, rel=1e-9)


def _discrete_champer_with_fertility(name: str, fertility: float) -> object:
    """Discrete population anchored at 2 * (K / 2) via the Champer route."""
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_autosomal_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={"female": {"W|W": K / 2}, "male": {"W|W": K / 2}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .build()
    )
    population.params.tensor_write(
        "fertility", np.array([0.0, fertility], dtype=np.float64)
    )
    population.update().competition(expected_num_new_adult_females=K / 2)
    return population


def test_discrete_champer_anchor_ignores_the_fertility_tensor() -> None:
    """The anchor's egg count must be derived with the tick's fertility rule.

    ``expected_num_new_adult_females`` is converted into
    ``external_expected_eggs`` by ``compute_expected_eggs_from_females``, which
    still reads the stored per-age weight verbatim.  On a discrete draft the
    tick reads no per-age fertility at all, so the derived override overstates
    production and the realised equilibrium leaves the declared anchor: the
    legal sequence below settles at 5000 (2.5 K) instead of 2000.
    """
    population = _discrete_champer_with_fertility("disc_champer_fert", 0.5)
    population.run(400)
    assert _total(population) == pytest.approx(K, rel=1e-9)


def test_age_structured_champer_anchor_clamps_the_fertility_weight() -> None:
    """The age path clamps the weight in the anchor derivation too.

    The age-structured tick consumes ``clamp01(fertility)``, so an
    out-of-domain stored 2.0 must yield the same anchor as the neutral weight.
    The derivation reads it raw, doubling the declared egg total and settling
    the population at 500 instead of 2000.
    """
    population = (
        nt.AgeStructuredPopulation.setup(
            species=_autosomal_species("age_champer_fert"),
            name="age_champer_fert",
            stochastic=False,
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"W|W": {1: K / 2}},
                "male": {"W|W": {1: K / 2}},
            }
        )
        .survival(
            female_age_based_survival=[1.0, S_F0], male_age_based_survival=[1.0, S_M0]
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0],
            eggs_per_female=10.0,
            sex_ratio=0.5,
        )
        .competition(
            carrying_capacity=K,
            low_density_growth_rate=3.0,
            growth_mode="beverton_holt",
        )
        .build()
    )
    population.params.tensor_write("fertility", np.array([0.0, 2.0], dtype=np.float64))
    population.update().competition(expected_num_new_adult_females=K / 2)
    population.run(600)
    assert _total(population) == pytest.approx(K, rel=1e-9)
