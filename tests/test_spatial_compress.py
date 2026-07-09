"""Numerical-verification tests for spatial compress.

Every assertion satisfies at least one of:
- Exact value (checklist #1): n_ztypes, state counts, genotype indices
- Domain invariant: migration conservation, no NaN, no negatives
- Structural: registry shared by reference
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.spatial.topology import SquareGrid, build_adjacency_matrix


def _make_sp(name: str) -> nt.Species:
    """Species with unique name to avoid cache collision across tests."""
    return nt.Species.from_dict(
        name, {"c1": {"l1": ["A", "a"]}},
        unordered=True, gamete_labels=["default"],
    )


def _make_drive() -> nt.HomingDrive:
    return nt.HomingDrive(
        name="d", drive_allele="A", target_allele="a",
        resistance_allele="a", drive_conversion_rate=1.0,
    )


# Canonical indices for unordered two-allele species: AA=0, Aa=1, aa=2.
# (A|A@default, A|a@default, a|a@default — unordered canonical ordering)
_IDX_AA = 0
_IDX_AA_RECESSIVE = 2


# ── Homo: compress shrinks registry ─────────────────────────────────────────────


class TestHomogeneousCompressExact:
    """Compress in homogeneous spatial builds produces exact n_ztypes.

    All tests use unique species names to avoid Species.from_dict cache
    collisions across test classes.
    """

    def test_compress_drive_prunes_unreachable(self):
        """From A|a with drive, BFS reaches exactly {AA, Aa} → n_ztypes=2."""
        sp = nt.Species.from_dict(
            "hc_drive_prune", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = _make_drive()
        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"A|a": 5000}, "male": {"A|a": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).build()

        reg = pop.demes[0].index_registry
        # Drive converts a→A: from Aa BFS reaches AA, Aa only.
        assert reg.n_ztypes == 2
        # aa must NOT exist.
        with pytest.raises(KeyError):
            reg.ztype_index(sp.get_genotype_from_str("a|a"), "default")
        # AA at index 0 (canonical), Aa at index 1.
        assert reg.ztype_index(sp.get_genotype_from_str("A|A"), "default") == 0
        assert reg.ztype_index(sp.get_genotype_from_str("A|a"), "default") == 1
        # Registry shared by reference.
        assert pop.demes[1].index_registry is reg

    def test_compress_no_drive_reaches_all(self):
        """Without drive, BFS from A|a reaches all 3 genotypes."""
        sp = nt.Species.from_dict(
            "hc_nodrive", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"A|a": 5000}, "male": {"A|a": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).build()

        reg = pop.demes[0].index_registry
        # A|a produces A and a gametes → BFS reaches all 3.
        assert reg.n_ztypes == 3
        # Canonical indices: AA=0, Aa=1, aa=2 (unordered).
        assert reg.ztype_index(sp.get_genotype_from_str("A|A"), "default") == 0
        assert reg.ztype_index(sp.get_genotype_from_str("A|a"), "default") == 1
        assert reg.ztype_index(sp.get_genotype_from_str("a|a"), "default") == 2

    def test_initial_seed_only_reaches_itself(self):
        """From AA alone, BFS only reaches AA → n_ztypes=1."""
        sp = nt.Species.from_dict(
            "hc_aa_only_v2", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        # Explicit unordered=True to avoid cache collision.
        full_n = len(sp.get_all_genotypes(unordered=True))
        assert full_n == 3

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).build()

        # From AA alone, BFS only reaches AA.
        assert pop.demes[0].index_registry.n_ztypes == 1
        # Only AA exists.
        assert pop.demes[0].index_registry.ztype_index(
            sp.get_genotype_from_str("A|A"), "default",
        ) == 0
        with pytest.raises(KeyError):
            pop.demes[0].index_registry.ztype_index(
                sp.get_genotype_from_str("A|a"), "default",
            )


# ── Hetero: union seeds produce compatible registries ───────────────────────────


class TestHeterogeneousUnionSeeds:
    """Union seeds guarantee all groups have same compressed registry.

    All tests use unique species names to avoid Species.from_dict cache
    collisions.
    """

    def test_two_demes_same_registry_size(self):
        """Deme A|a + deme A|A → both have n_ztypes=3, AA in both."""
        sp = nt.Species.from_dict(
            "hu_2deme", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count=nt.batch_setting([
                {"female": {"A|a": 100}, "male": {"A|a": 100}},
                {"female": {"A|A": 100}, "male": {"A|A": 100}},
            ]),
        ).competition(
            carrying_capacity=200, juvenile_growth_mode=nt.NO_COMPETITION,
        ).build()

        reg0 = pop.demes[0].index_registry
        reg1 = pop.demes[1].index_registry

        # Invariant: same n_ztypes (union seeds prevent divergence).
        assert reg0.n_ztypes == reg1.n_ztypes == 3

        # Genotype indices must be identical across demes.
        sp_gt = sp.get_genotype_from_str
        assert reg0.ztype_index(sp_gt("A|A"), "default") == 0
        assert reg1.ztype_index(sp_gt("A|A"), "default") == 0
        assert reg0.ztype_index(sp_gt("A|a"), "default") == 1
        assert reg1.ztype_index(sp_gt("A|a"), "default") == 1
        assert reg0.ztype_index(sp_gt("a|a"), "default") == 2
        assert reg1.ztype_index(sp_gt("a|a"), "default") == 2

    def test_initial_state_counts_conserved(self):
        """Total individuals after build = exact sum of per-deme initial counts."""
        sp = nt.Species.from_dict(
            "hu_counts", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count=nt.batch_setting([
                {"female": {"A|a": 100}, "male": {"A|a": 100}},   # 200
                {"female": {"A|A": 300}, "male": {"A|A": 300}},   # 600
            ]),
        ).competition(
            carrying_capacity=1000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).build()

        # Exact initial counts — confirm compression didn't alter them.
        assert pop.demes[0].state.individual_count.sum() == 200.0
        assert pop.demes[1].state.individual_count.sum() == 600.0
        # No NaN, no negatives.
        for d in pop.demes:
            s = d.state.individual_count
            assert not np.any(np.isnan(s))
            assert np.all(s >= 0)


# ── Compress + migration ────────────────────────────────────────────────────────


class TestCompressMigrationInvariants:
    """Compressed registries are compatible with cross-deme migration."""

    @pytest.mark.numba_off
    def test_migration_conservation(self, simple_species):
        """Migration preserves total individuals (conservation invariant)."""
        sp = simple_species

        def _make_deme(name: str, count: float) -> nt.AgeStructuredPopulation:
            return nt.AgeStructuredPopulation.setup(
                species=sp, name=name, stochastic=False, compress=True,
            ).age_structure(n_ages=4, new_adult_age=1).initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, count, 0.0, 0.0]},
                    "male": {"WT|WT": [0.0, count, 0.0, 0.0]},
                },
            ).survival(
                female_age_based_survival=[1.0, 1.0, 1.0, 0.0],
                male_age_based_survival=[1.0, 1.0, 1.0, 0.0],
            ).reproduction(
                female_age_based_mating_rate=[0.0, 0.0, 0.0, 0.0],
                male_age_based_mating_rate=[0.0, 0.0, 0.0, 0.0],
                eggs_per_female=0.0,
            ).competition(
                juvenile_growth_mode="logistic",
                expected_num_new_adult_females=100,
                old_juvenile_carrying_capacity=200.0,
            ).build()

        demes = [_make_deme("d0", 100.0), _make_deme("d1", 0.0),
                 _make_deme("d2", 0.0), _make_deme("d3", 0.0)]

        shared = demes[0].export_config()
        for d in demes[1:]:
            d.import_config(shared._replace())

        adjacency = build_adjacency_matrix(
            SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
            row_normalize=True,
        )
        spatial = nt.SpatialPopulation(
            demes=demes, adjacency=adjacency, migration_rate=0.5,
        )

        # Pre-migration: exactly 200 total (2 sexes × 100 each).
        pre = sum(float(d.state.individual_count.sum()) for d in spatial.demes)
        assert pre == 200.0

        spatial.run_tick()
        after = [float(d.state.individual_count.sum()) for d in spatial.demes]

        # Conservation invariant.
        assert sum(after) == pytest.approx(200.0, abs=1e-6)

        # Non-trivial migration: source lost, sinks gained.
        assert after[0] < 200.0
        assert all(a >= 0 for a in after)
        assert all(a > 0 for a in after[1:])

    def test_compress_uncompress_same_initial_total(self):
        """Compressed and uncompressed builds have same initial state total."""
        sp = nt.Species.from_dict(
            "hu_same_total", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )

        def _build(**kw):
            return nt.SpatialPopulation.builder(
                species=sp, n_demes=1, pop_type="discrete_generation",
            ).setup(stochastic=False, **kw).initial_state(
                individual_count={"female": {"A|a": 5000}, "male": {"A|a": 5000}},
            ).competition(
                carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
            ).build()

        pop_u = _build(compress=False)
        pop_c = _build(compress=True)

        # After build, total individuals are identical (compress only
        # changes array dimensions, not values).
        u_sum = pop_u.demes[0].state.individual_count.sum()
        c_sum = pop_c.demes[0].state.individual_count.sum()
        assert u_sum == 10000.0
        assert c_sum == 10000.0

        # Sanity: no NaN, no negatives in compressed state.
        assert not np.any(np.isnan(pop_c.demes[0].state.individual_count))
        assert np.all(pop_c.demes[0].state.individual_count >= 0)
