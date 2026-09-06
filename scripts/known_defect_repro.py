"""Red-light repros for the known defects R1-R5 (RUST_ONLY_REFACTOR_PLAN section 3).

Each function asserts the **correct** behavior; every assertion currently
FAILS on this tree because the corresponding defect is present.  This
script is construction scaffolding for stage S0, not a pytest module:
per the plan, "the red-light repros are construction preparation, not a
code delivery that can request APPROVED".  When the owning stage lands
(S2 absorbs R3/R4/R5, S3 absorbs R1/R2), each repro turns green and is
promoted into the regular pytest suite together with its fix in the
same batch.

Run: ``python scripts/known_defect_repro.py`` prints one line per
defect with PASS (defect gone) / FAIL (defect present) plus evidence.
"""

from __future__ import annotations

import argparse
import traceback
from typing import Callable

import numpy as np

import natal as nt

# ── shared fixtures ───────────────────────────────────────────────────────────


def _two_allele_species(name: str) -> nt.Species:
    """Return a fresh two-allele species with a unique singleton name."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _ring_adjacency(n: int) -> np.ndarray:
    """Return the dense adjacency of an open chain 0-1-...-(n-1)."""
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        m[i, i + 1] = 1.0
        m[i + 1, i] = 1.0
    return m


# ── R1: spatial Rust rebuilds the per-deme RNG from `seed ^ deme_id` every tick
#
# Discriminating probe: population A runs two ticks; population B is rebuilt
# with the same seed and receives A's tick-1 state exactly (via import_state).
# A's tick 2 and B's tick 1 then share the same input.  A healthy engine holds
# one persistent per-deme stream, so A's tick 2 draws from an already-advanced
# stream while B starts fresh — the outputs must differ.  The defect reseeds
# `new_rng(seed ^ deme)` at every tick, so both sides restart the identical
# stream on the identical input and the outputs match bitwise.


def repro_r1() -> None:
    """Assert the spatial Rust RNG advances across ticks (currently fails)."""
    species = _two_allele_species("ReproR1Species")

    def build(name: str, seed: int) -> nt.SpatialPopulation:
        pop = (
            nt.SpatialPopulation.builder(
                species, n_demes=4, pop_type="age_structured"
            )
            .setup(name=name, stochastic=True)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count=nt.batch_setting(
                    [
                        {
                            "female": {"WT|WT": [0.0, 100.0, 0.0]},
                            "male": {"WT|WT": [0.0, 100.0, 0.0]},
                        },
                    ]
                    * 4
                )
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0, 0.0],
                male_age_based_mating_rate=[0.0, 1.0, 0.0],
                eggs_per_female=4.0,
            )
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.0],
                male_age_based_survival=[1.0, 0.9, 0.0],
            )
            .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
            .migration(adjacency=_ring_adjacency(4), migration_rate=0.25)
            .build()
        )
        pop.enable_rust_backend(seed=seed)
        return pop

    a = build("ReproR1A", seed=42)
    a.run(1)
    ind = [d.state.individual_count.copy() for d in a.demes]
    sperm = [d.state.sperm_storage.copy() for d in a.demes]
    a.run(1)
    final_a = np.stack([d.state.individual_count for d in a.demes])

    b = build("ReproR1B", seed=42)
    for deme, i, s in zip(b.demes, ind, sperm):
        deme.import_state(
            {"n_tick": 1, "individual_count": i, "sperm_storage": s}
        )
    b.run(1)
    final_b = np.stack([d.state.individual_count for d in b.demes])

    assert not np.array_equal(
        final_a, final_b
    ), "R1: A(two ticks) equals B(one tick from A's tick-1 state) bitwise — the per-deme RNG is reseeded every tick instead of advancing"


# ── R2: discrete spatial Rust backend is built with HookProgram=None
#
# Discriminating probe: the same declarative halve-K hook runs on the plain
# discrete population and on a discrete spatial population.  The plain path
# applies it; the discrete-spatial Rust path installs no hook program, so K
# never changes.


def repro_r2() -> None:
    """Assert declarative hooks run on the discrete spatial Rust path (currently fails)."""
    species = _two_allele_species("ReproR2Species")

    @nt.hook(event="late")
    def halve_k() -> list[nt.HookOp]:
        return [nt.Op.set_param("carrying_capacity", "K * 0.5")]

    spatial = (
        nt.SpatialPopulation.builder(
            species, n_demes=4, pop_type="discrete_generation"
        )
        .setup(name="ReproR2Spatial", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": 100.0},
                        "male": {"WT|WT": 100.0},
                    },
                ]
                * 4
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=10000.0, low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(4), migration_rate=0.0)
        .hooks(halve_k)
        .build()
    )
    spatial.enable_rust_backend(seed=7)
    spatial.run(2)
    ks = [d.params.carrying_capacity for d in spatial.demes]

    plain = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="ReproR2Plain", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 100.0},
                "male": {"WT|WT": 100.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=10000.0, low_density_growth_rate=2.0)
        .hooks(halve_k)
        .build()
    )
    plain.run(2)

    assert all(k == 2500.0 for k in ks), (
        "R2: discrete spatial Rust path skipped the declarative hook — "
        f"K stayed at {ks} while the plain path reached {plain.params.carrying_capacity}"
    )


# ── R3: public restore_checkpoint only restores counts/tick, not ecology/RNG


def repro_r3() -> None:
    """Assert restore_checkpoint rolls ecology back to the checkpointed tick (currently fails)."""
    species = _two_allele_species("ReproR3Species")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="ReproR3Pop", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .record_history(mode="raw")
        .build()
    )
    pop.run(3, record_every=1)
    pop.update().competition(carrying_capacity=4321.0)
    pop.restore_checkpoint(1)

    assert pop.params.carrying_capacity == 100000.0, (
        "R3: restore_checkpoint(1) kept the later K="
        f"{pop.params.carrying_capacity} instead of the checkpointed 100000.0"
    )


# ── R4: the eight Blueprint ndarrays are writable
#
# A NamedTuple freezes field binding, not array contents: any holder can
# mutate the engine's model arrays through the shared references.


def repro_r4() -> None:
    """Assert materialized Blueprint arrays reject writes (currently fails)."""
    from natal.contracts.materialize import materialize

    species = _two_allele_species("ReproR4Species")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="ReproR4Pop", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )
    mat = materialize(pop.config)
    blueprint = mat.blueprint

    array_fields = [
        "adult_ages",
        "female_only_by_sex_chrom",
        "male_only_by_sex_chrom",
        "initial_individual_count",
        "initial_sperm_storage",
        "migration_indptr",
        "migration_dest_idx",
        "migration_weights",
    ]
    writable = [
        field for field in array_fields
        if isinstance(getattr(blueprint, field), np.ndarray)
        and getattr(blueprint, field).flags.writeable
    ]

    assert not writable, (
        f"R4: Blueprint ndarray fields are writable: {writable} — external "
        "holders can mutate the engine's frozen model arrays in place"
    )


# ── R5: pop.state returns the live internal state container


def repro_r5() -> None:
    """Assert mutating the returned state cannot reach the engine (currently fails)."""
    species = _two_allele_species("ReproR5Species")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="ReproR5Pop", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )
    before = pop.state.individual_count.copy()
    pop.state.individual_count[0, 1, 0] = 999.0

    assert np.array_equal(
        pop.state.individual_count, before
    ), "R5: writing through the returned state container mutated the engine's live arrays"


# ── runner ────────────────────────────────────────────────────────────────────

REPROS: dict[str, Callable[[], None]] = {
    "R1": repro_r1,
    "R2": repro_r2,
    "R3": repro_r3,
    "R4": repro_r4,
    "R5": repro_r5,
}

OWNING_STAGE = {
    "R1": "S3 (persistent per-deme RNG + unified Program)",
    "R2": "S3 (one Program for all models)",
    "R3": "S2/S4 (public restore wired to the Rust checkpoint)",
    "R4": "S2 (frozen Blueprint inside the Rust session)",
    "R5": "S2 (snapshots instead of live containers)",
}


def main(only: list[str] | None = None) -> int:
    """Run the red-light repros and print per-defect evidence.

    Args:
        only: Optional subset of defect ids to run.

    Returns:
        Process exit code: 0 always — this is an evidence report, not a gate.
    """
    selected = only or list(REPROS)
    print("Known-defect red-light repros (S0 scaffolding, not a pytest module)")
    print("=" * 72)
    for defect_id in selected:
        func = REPROS[defect_id]
        try:
            func()
        except AssertionError as exc:
            print(f"[FAIL] {defect_id} defect present — fixed by {OWNING_STAGE[defect_id]}")
            print(f"       {exc}")
        except Exception:  # noqa: BLE001  # evidence report must survive any error
            print(f"[ERROR] {defect_id} repro crashed — treat as defect present")
            traceback.print_exc()
        else:
            print(f"[PASS] {defect_id} defect gone — promote this repro into pytest")
    print("=" * 72)
    print("Each [FAIL] line is the S0 red-light evidence for the plan ledger;")
    print("the owning stage must turn it green and promote it into the suite.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", choices=sorted(REPROS), help="Run a subset of defects"
    )
    args = parser.parse_args()
    raise SystemExit(main(only=args.only))
