"""Adversarial probes for the kernel config-snapshot removal refactor.

The refactor (removal of ``kernels/config.rs``) made the lifecycle kernels
read ``&Blueprint``/``&EcologyParams``/``&GeneticsTensors`` plus a deme index
directly instead of an assembled snapshot.  These probes attack the paths the
phase-0 digest scenarios leave thin:

1. Equilibrium-metric freshness under a same-tick early-hook ``set_param``
   write in a compensatory (logistic) model — the retired config layer
   recomputed the metrics at every post-commit re-assembly; the live-column
   kernels must observe the same values.
2. End-to-end per-deme heterogeneous ecology (scalar AND vector columns,
   female/male asymmetric) against panmictic twin oracles — a wrong deme
   index or segment stride cannot pass when every deme carries distinct
   ecology.
3. EcoCtx (hooked, local-copy) vs columns (hook-free) stage-source parity:
   a value-preserving ``set_param`` hook must not change a stochastic
   trajectory for the same seed.
4. Sex-chromosome stochastic rerun determinism (integer and continuous
   sampling) through the re-plumbed genetics tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import natal as nt  # noqa: E402
from natal.frontend.hooks.entry.declarative import Op  # noqa: E402
from natal.frontend.spatial.builder import batch_setting  # noqa: E402

try:
    from natal.backends.rust.rust_backend import rust_backend_available

    RUST_AVAILABLE = rust_backend_available()
except Exception:  # pragma: no cover - import guard for unbuilt extensions
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="natal._engine_rs is not built"
)

_SPECIES_COUNTER = 0


def _fresh_species(prefix: str) -> nt.Species:
    """Return a uniquely named two-allele species (registry cache safe)."""
    global _SPECIES_COUNTER
    _SPECIES_COUNTER += 1
    return nt.Species.from_dict(
        name=f"{prefix}Species{_SPECIES_COUNTER}",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _fresh_xy_species(prefix: str) -> nt.Species:
    """Return a uniquely named X/Y sex-chromosome species."""
    global _SPECIES_COUNTER
    _SPECIES_COUNTER += 1
    return nt.Species.from_dict(
        name=f"{prefix}XYSpecies{_SPECIES_COUNTER}",
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y"]}},
        },
        unordered=False,
    )


def _pick_xy_parents(species: nt.Species) -> tuple[object, object]:
    """Return one female (X/X) and one male (X/Y) genotype of *species*."""

    def _has(haploid: object, chromosome: object) -> bool:
        try:
            haploid.get_haplotype_for_chromosome(chromosome)
            return True
        except ValueError:
            return False

    chrom_x = species.get_chromosome("chrX")
    chrom_y = species.get_chromosome("chrY")
    female = male = None
    for genotype in species.get_all_genotypes():
        maternal = genotype.maternal
        paternal = genotype.paternal
        if _has(maternal, chrom_x) and not _has(maternal, chrom_y):
            if _has(paternal, chrom_x) and not _has(paternal, chrom_y):
                female = genotype
            if not _has(paternal, chrom_x) and _has(paternal, chrom_y):
                male = genotype
    assert female is not None and male is not None
    return female, male


def _age_population(
    species: nt.Species,
    name: str,
    *,
    stochastic: bool = False,
    continuous: bool = False,
    hooks=None,
    growth_mode: int | str = 2,
    k: float = 500.0,
    r: float = 3.0,
    eggs: float = 10.0,
    female_survival=0.9,
    male_survival=0.85,
    female_mating=1.0,
    male_mating=0.8,
    initial_state=None,
):
    """Deterministic-by-default age-structured population with sperm storage."""
    if initial_state is None:
        initial_state = {
            "female": {"A|A": 200.0, "A|B": 100.0},
            "male": {"A|A": 150.0, "A|B": 150.0},
        }
    builder = (
        nt.AgeStructuredPopulation.setup(
            species, name=name, stochastic=stochastic,
            continuous_sampling=continuous,
        )
        .initial_state(individual_count=initial_state)
        .reproduction(
            eggs_per_female=eggs,
            sex_ratio=0.5,
            female_age_based_mating_rate=female_mating,
            male_age_based_mating_rate=male_mating,
            age_based_reproduction_rate=1.0,
            female_age_based_fertility=1.0,
            fixed_egg_count=True,
        )
        .survival(
            female_age_based_survival=female_survival,
            male_age_based_survival=male_survival,
        )
        .competition(
            juvenile_growth_mode=growth_mode, carrying_capacity=k,
            low_density_growth_rate=r,
        )
    )
    if hooks is not None:
        builder = hooks(builder)
    return builder.build()


# ---------------------------------------------------------------------------
# 1. Equilibrium-metric freshness under a same-tick set_param write
# ---------------------------------------------------------------------------


def test_early_hook_k_write_refreshes_logistic_metrics_same_tick() -> None:
    """A tick-1 early-hook K write must feed tick-1 survival immediately.

    Oracle: a manual ``params`` write between run calls applies to the whole
    next tick.  Because ``carrying_capacity`` is consumed only by the
    survival stage's density regulation (the first hook and reproduction
    never read it), the declarative early-hook model and the manual model
    must agree bitwise — but only if the write is committed at the event
    boundary AND the equilibrium metrics (which derive from K) are evaluated
    from the committed columns, not a stale batch-entry snapshot.
    """
    species = _fresh_species("Kw")
    hooked = _age_population(
        species, "k_write_hooked", growth_mode=2, k=500.0, r=3.0,
        hooks=lambda builder: builder.hooks(
            [Op.set_param("carrying_capacity", 150.0, start=1)],
            event="early", name="k_write",
        ),
    )
    species_manual = _fresh_species("Kw")
    manual = _age_population(species_manual, "k_write_manual", growth_mode=2, k=500.0, r=3.0)
    species_frozen = _fresh_species("Kw")
    frozen = _age_population(species_frozen, "k_write_frozen", growth_mode=2, k=500.0, r=3.0)

    # One single batch call: a stale batch-entry config would keep K=500
    # metrics for every tick of this run.
    hooked.run(6, record_every=0)

    manual.run(1, record_every=0)
    manual.params.carrying_capacity = 150.0
    manual.run(5, record_every=0)

    frozen.run(6, record_every=0)

    np.testing.assert_array_equal(
        hooked.state.individual_count,
        manual.state.individual_count,
        err_msg="early-hook K write must match a manual mid-run write bitwise",
    )
    np.testing.assert_array_equal(
        hooked.state.sperm_storage,
        manual.state.sperm_storage,
        err_msg="sperm plane must agree with the manual mid-run write",
    )
    # Discrimination guard: the K=150 dynamics must actually differ from the
    # frozen-K trajectory (otherwise this probe asserts nothing).
    assert not np.array_equal(
        hooked.state.individual_count, frozen.state.individual_count
    ), "probe is not discriminating: K write did not change the trajectory"
    assert hooked.params.carrying_capacity == 150.0


# ---------------------------------------------------------------------------
# 2. Heterogeneous per-deme ecology vs panmictic twin oracles
# ---------------------------------------------------------------------------

_N_DEMES_PROBE = 3

# Per-deme ecology: every deme differs in every scalar and vector channel,
# including female-vs-male asymmetric vectors, so any wrong deme index or
# wrong (sex, age) stride inside a segment must change some deme's numbers.
_PROBE_FEMALE_SURVIVAL = [
    np.array([0.50, 0.95], dtype=np.float64),
    np.array([0.65, 0.70], dtype=np.float64),
    np.array([0.80, 0.55], dtype=np.float64),
]
_PROBE_MALE_SURVIVAL = [
    np.array([0.45, 0.90], dtype=np.float64),
    np.array([0.75, 0.60], dtype=np.float64),
    np.array([0.55, 0.85], dtype=np.float64),
]
_PROBE_FEMALE_MATING = [
    np.array([0.0, 0.95], dtype=np.float64),
    np.array([0.10, 0.70], dtype=np.float64),
    np.array([0.0, 0.50], dtype=np.float64),
]
_PROBE_MALE_MATING = [
    np.array([0.0, 0.80], dtype=np.float64),
    np.array([0.05, 0.60], dtype=np.float64),
    np.array([0.0, 0.40], dtype=np.float64),
]
_PROBE_K = [500.0, 320.0, 180.0]
_PROBE_R = [3.0, 2.2, 4.5]
_PROBE_EGGS = [12.0, 9.0, 15.0]
_PROBE_INITIAL = [
    {"female": {"A|A": 200.0, "A|B": 60.0}, "male": {"A|A": 120.0, "A|B": 90.0}},
    {"female": {"A|A": 80.0, "A|B": 160.0}, "male": {"A|A": 140.0, "A|B": 40.0}},
    {"female": {"A|A": 150.0, "A|B": 20.0}, "male": {"A|A": 30.0, "A|B": 130.0}},
]


def _heterogeneous_spatial(species: nt.Species, name: str, *, stochastic: bool, hooks=None):
    """3-deme spatial population; identity adjacency, zero migration.

    Every ecology channel is per-deme distinct (scalars, per-sex survival
    and mating vectors), so the demes evolve independently and each must
    match a panmictic twin built from its own values.
    """
    builder = (
        nt.SpatialPopulation.builder(species, n_demes=_N_DEMES_PROBE)
        .setup(name=name, stochastic=stochastic)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count=batch_setting([dict(d) for d in _PROBE_INITIAL]))
        .survival(
            female_age_based_survival=batch_setting([v.copy() for v in _PROBE_FEMALE_SURVIVAL]),
            male_age_based_survival=batch_setting([v.copy() for v in _PROBE_MALE_SURVIVAL]),
        )
        .reproduction(
            eggs_per_female=batch_setting(list(_PROBE_EGGS)),
            sex_ratio=0.5,
            female_age_based_mating_rate=batch_setting([v.copy() for v in _PROBE_FEMALE_MATING]),
            male_age_based_mating_rate=batch_setting([v.copy() for v in _PROBE_MALE_MATING]),
            age_based_reproduction_rate=np.array([0.0, 0.9]),
            female_age_based_fertility=np.array([0.0, 1.0]),
        )
        .competition(
            juvenile_growth_mode="logistic",
            carrying_capacity=batch_setting(list(_PROBE_K)),
            low_density_growth_rate=batch_setting(list(_PROBE_R)),
        )
    )
    if hooks is not None:
        builder = hooks(builder)
    return builder.migration(migration_rate=0.0).build()


def _panmictic_twin(species: nt.Species, name: str, deme: int, *, stochastic: bool):
    """Panmictic twin carrying deme *deme*'s ecology and initial state."""
    initial = _PROBE_INITIAL[deme]
    return (
        nt.AgeStructuredPopulation.setup(species, name=name, stochastic=stochastic)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": dict(initial["female"]),
            "male": dict(initial["male"]),
        })
        .reproduction(
            eggs_per_female=_PROBE_EGGS[deme],
            sex_ratio=0.5,
            female_age_based_mating_rate=_PROBE_FEMALE_MATING[deme].tolist(),
            male_age_based_mating_rate=_PROBE_MALE_MATING[deme].tolist(),
            age_based_reproduction_rate=np.array([0.0, 0.9]),
            female_age_based_fertility=np.array([0.0, 1.0]),
        )
        .survival(
            female_age_based_survival=_PROBE_FEMALE_SURVIVAL[deme].tolist(),
            male_age_based_survival=_PROBE_MALE_SURVIVAL[deme].tolist(),
        )
        .competition(
            juvenile_growth_mode="logistic",
            carrying_capacity=_PROBE_K[deme],
            low_density_growth_rate=_PROBE_R[deme],
        )
        .build()
    )


def test_heterogeneous_spatial_demes_match_panmictic_twins() -> None:
    """Every deme's trajectory must equal its own panmictic twin bitwise.

    With zero migration the demes are independent, and in deterministic
    mode the trajectory is a pure function of (initial state, ecology
    columns).  Reading any other deme's column — or a mis-strided segment
    inside the (2, A) survival/mating vectors — breaks at least one twin
    parity.  The logistic mode additionally exercises the per-deme
    equilibrium-metric derivation inside each deme's density regulation.
    """
    species = _fresh_species("Het")
    spatial = _heterogeneous_spatial(species, "het_oracle_spatial", stochastic=False)
    spatial.run(8, record_every=0)

    for deme in range(_N_DEMES_PROBE):
        twin_species = _fresh_species("Het")
        twin = _panmictic_twin(
            twin_species, f"het_oracle_twin_{deme}", deme, stochastic=False
        )
        twin.run(8, record_every=0)
        np.testing.assert_array_equal(
            spatial.deme(deme).state.individual_count,
            twin.state.individual_count,
            err_msg=f"deme {deme} diverged from its panmictic twin",
        )
        np.testing.assert_array_equal(
            spatial.deme(deme).state.sperm_storage,
            twin.state.sperm_storage,
            err_msg=f"deme {deme} sperm plane diverged from its twin",
        )

    # Discrimination guard: demes must not all share one trajectory (the
    # heterogeneous ecology has to matter), otherwise the twin parities
    # would hold even for a constant deme-0 read.
    states = [spatial.deme(d).state.individual_count.copy() for d in range(_N_DEMES_PROBE)]
    assert not np.array_equal(states[0], states[1])
    assert not np.array_equal(states[1], states[2])


# ---------------------------------------------------------------------------
# 3. EcoCtx (hooked local-copy) vs columns (hook-free) stage-source parity
# ---------------------------------------------------------------------------


def test_value_preserving_set_param_hook_leaves_trajectory_bitwise() -> None:
    """A hooked deme (EcoCtx local copy, deme 0 reads) must match a hook-free
    deme (direct session-column reads at its own index) bitwise.

    The hook writes ``K * 1.0`` (a value-preserving expression, so no
    journal row and no parameter change) and — being a pure ``set_param``
    op — consumes no RNG draws.  Same seed, same ops: the only acceptable
    difference between the two programs is none at all.  Any divergence
    between the EcoCtx stage sources and the columns stage sources (wrong
    local-copy cut, deme 0 vs deme-id mismatch, stale scratch) shows up as
    a trajectory difference in stochastic mode.
    """
    def _build(name: str, *, hooked: bool):
        species = _fresh_species("Vp")
        hooks = None
        if hooked:
            def hooks(builder):  # noqa: F811 - rebinding keeps _heterogeneous_spatial generic
                return builder.hooks(
                    [Op.set_param("carrying_capacity", "K * 1.0", every=1)],
                    event="early", name="k_identity",
                )
        return _heterogeneous_spatial(
            species, name, stochastic=True, hooks=hooks
        )

    plain = _build("vp_plain", hooked=False)
    hooked = _build("vp_hooked", hooked=True)

    plain.run(6, record_every=0)
    hooked.run(6, record_every=0)

    for deme in range(_N_DEMES_PROBE):
        np.testing.assert_array_equal(
            plain.deme(deme).state.individual_count,
            hooked.deme(deme).state.individual_count,
            err_msg=f"hooked deme {deme} diverged from the hook-free run",
        )
        np.testing.assert_array_equal(
            plain.deme(deme).state.sperm_storage,
            hooked.deme(deme).state.sperm_storage,
            err_msg=f"hooked deme {deme} sperm plane diverged",
        )
    # The identity write must not be journaled anywhere (old == new for
    # every deme): each deme's own audit log stays empty like the plain run.
    for deme in range(_N_DEMES_PROBE):
        assert list(hooked.demes[deme].params_log) == list(plain.demes[deme].params_log), (
            f"deme {deme} journaled the identity write"
        )


# ---------------------------------------------------------------------------
# 4. Sex-chromosome stochastic rerun determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuous", [False, True])
def test_sex_chromosome_stochastic_rerun_deterministic(continuous: bool) -> None:
    """Same seed, same ops: two fresh builds must agree bitwise.

    The sex-chromosome fertilize branch reads the blueprint flags and the
    ztype-compatibility genetics tables through the new direct plumbing;
    integer and continuous sampling both must stay deterministic under
    re-runs.
    """
    def _build(name: str):
        species = _fresh_xy_species("Sc")
        female_genotype, male_genotype = _pick_xy_parents(species)
        return _age_population(
            species,
            name,
            stochastic=True,
            continuous=continuous,
            growth_mode=3,
            k=400.0,
            female_survival=0.85,
            male_survival=0.8,
            initial_state={
                "female": {female_genotype: 200.0},
                "male": {male_genotype: 150.0},
            },
        )

    trajectories = []
    for run_index in range(2):
        pop = _build(f"sc_rerun_{run_index}")
        pop.run(5, record_every=0)
        trajectories.append(
            (pop.state.individual_count.copy(), pop.state.sperm_storage.copy())
        )

    np.testing.assert_array_equal(
        trajectories[0][0], trajectories[1][0],
        err_msg="sex-chromosome counts differ across same-seed reruns",
    )
    np.testing.assert_array_equal(
        trajectories[0][1], trajectories[1][1],
        err_msg="sex-chromosome sperm differs across same-seed reruns",
    )
    # Both sexes present keeps the sex-chromosome assignment branch live.
    assert trajectories[0][0].sum(axis=(0, 2))[1] > 0.0
