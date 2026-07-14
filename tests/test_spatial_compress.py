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

    def test_opposite_homozygotes_discover_heterozygote(self):
        """AA + aa seeds → BFS propagates to Aa via union gamete pool.

        Deme 0 starts with only A|A, deme 1 with only a|a.  Neither deme
        contains A|a in its initial state.  But union seeds {A|A, a|a}
        make BFS produce both A and a gametes, whose cross yields A|a.
        The compressed registry must include all three genotypes in both
        demes — otherwise migration of A|a individuals between demes
        would fail.
        """
        sp = nt.Species.from_dict(
            "hu_opposite", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count=nt.batch_setting([
                {"female": {"A|A": 100}, "male": {"A|A": 100}},
                {"female": {"a|a": 100}, "male": {"a|a": 100}},
            ]),
        ).competition(
            carrying_capacity=200, juvenile_growth_mode=nt.NO_COMPETITION,
        ).build()

        reg0 = pop.demes[0].index_registry
        reg1 = pop.demes[1].index_registry

        # Both demes must see all 3 genotypes.
        assert reg0.n_ztypes == reg1.n_ztypes == 3

        # Index mapping identical across demes.
        sp_gt = sp.get_genotype_from_str
        assert reg0.ztype_index(sp_gt("A|A"), "default") == 0
        assert reg1.ztype_index(sp_gt("A|A"), "default") == 0
        assert reg0.ztype_index(sp_gt("A|a"), "default") == 1
        assert reg1.ztype_index(sp_gt("A|a"), "default") == 1
        assert reg0.ztype_index(sp_gt("a|a"), "default") == 2
        assert reg1.ztype_index(sp_gt("a|a"), "default") == 2

    def test_preset_heterogeneity_combined_maps_protect_all(self):
        """Drive on deme A + no drive on deme B → aa survives in both.

        Deme 0 has drive (a→A conversion), deme 1 has no drive but
        starts with a|a.  Without combined modifier maps, deme 0's
        per-deme BFS would prune aa.  Combined maps sum drive-modified
        + Mendelian baseline → aa gamete reachable → aa protected.
        """
        sp = nt.Species.from_dict(
            "hu_preset", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = _make_drive()
        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count=nt.batch_setting([
                {"female": {"A|a": 100}, "male": {"A|a": 100}},
                {"female": {"a|a": 100}, "male": {"a|a": 100}},
            ]),
        ).competition(
            carrying_capacity=200, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(nt.batch_setting([drive, None])).build()

        reg0 = pop.demes[0].index_registry
        reg1 = pop.demes[1].index_registry
        assert reg0.n_ztypes == reg1.n_ztypes == 3
        sp_gt = sp.get_genotype_from_str
        assert reg0.ztype_index(sp_gt("a|a"), "default") == 2
        assert reg1.ztype_index(sp_gt("a|a"), "default") == 2


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


# ── Spatial + hook + compress ───────────────────────────────────────────────────


class TestSpatialHookCompress:
    """Hook genotype refs are auto-protected from compression in spatial models."""

    def test_homo_hook_protects_genotype_from_drive_pruning(self):
        """Drive prunes aa, but hook referencing aa keeps it at index 2."""
        sp = nt.Species.from_dict(
            "shc_drive", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = _make_drive()

        @nt.hook(event="first", priority=0)
        def release():
            return [nt.Op.add(genotypes="a|a", ages=1, sex="male", delta=10)]

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"A|a": 5000}, "male": {"A|a": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).hooks(release).build()

        reg = pop.demes[0].index_registry
        # Without hook: n_ztypes=2. With hook auto-resolution: 3.
        assert reg.n_ztypes == 3
        assert reg.ztype_index(sp.get_genotype_from_str("a|a"), "default") == 2
        assert pop.demes[1].index_registry is reg

    def test_hetero_hook_in_union_seeds(self):
        """Hook refs from all demes included in union seeds."""
        sp = nt.Species.from_dict(
            "shc_union", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )

        @nt.hook(event="first", priority=0)
        def release():
            return [nt.Op.add(genotypes="A|A", ages=1, sex="male", delta=10)]

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=2, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count=nt.batch_setting([
                {"female": {"a|a": 100}, "male": {"a|a": 100}},
                {"female": {"A|a": 100}, "male": {"A|a": 100}},
            ]),
        ).competition(
            carrying_capacity=200, juvenile_growth_mode=nt.NO_COMPETITION,
        ).hooks(release).build()

        r0 = pop.demes[0].index_registry
        r1 = pop.demes[1].index_registry
        assert r0.n_ztypes == r1.n_ztypes
        AA = sp.get_genotype_from_str("A|A")
        assert r0.ztype_index(AA, "default") == r1.ztype_index(AA, "default") == 0

    def test_selector_hook_protects_genotype(self):
        """Selector hook protects aa at exact index 2."""
        sp = nt.Species.from_dict(
            "shc_sel", {"c1": {"l1": ["A", "a"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = _make_drive()

        @nt.hook(event="first", priority=0, selectors={"target": "a|a"})
        def count_aa(state, config, target, deme_id=-1):
            _ = target

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"A|a": 5000}, "male": {"A|a": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).hooks(count_aa).build()

        reg = pop.demes[0].index_registry
        assert reg.n_ztypes == 3
        assert reg.ztype_index(sp.get_genotype_from_str("a|a"), "default") == 2


# ============================================================================
# Hook pattern formats — auto-collection for compression BFS seeds
# ============================================================================


class TestSpatialHookPatternFormats:
    """Hook auto-collection handles genotype patterns, not just plain strings."""

    def test_declarative_hook_wildcard_pattern(self):
        """Op.add with 'Dr|*' wildcard protects all Dr-carrying genotypes."""
        sp = nt.Species.from_dict(
            "shc_wild", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = nt.HomingDrive(
            name="d", drive_allele="Dr", target_allele="WT",
            resistance_allele="R2", drive_conversion_rate=1.0,
        )

        @nt.hook(event="first", priority=0)
        def release():
            return [nt.Op.add(genotypes="Dr|*", ages=1, sex="male", delta=10)]

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).hooks(release).build()

        reg = pop.demes[0].index_registry
        # Dr|Dr should exist even though it's not in initial state
        dr_dr = sp.get_genotype_from_str("Dr|Dr")
        assert reg.ztype_index(dr_dr, "default") >= 0

    def test_declarative_hook_slab_suffix(self):
        """Op.add with @slab suffix survives compression (regression for #34 fix)."""
        sp = nt.Species.from_dict(
            "shc_slab", {"c1": {"l1": ["WT", "Dr"]}},
            unordered=True, somatic_labels=["S", "E"],
            gamete_labels=["default"],
        )

        @nt.hook(event="first", priority=0)
        def release():
            return [nt.Op.add(genotypes="Dr|WT@E", ages=1, sex="male", delta=10)]

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).hooks(release).build()

        reg = pop.demes[0].index_registry
        # Dr|WT@E must be resolvable
        dr_wt = sp.get_genotype_from_str("Dr|WT")
        assert reg.ztype_index(dr_wt, "E") >= 0


class TestSpatialSelectorHookPatterns:
    """Selector hook auto-collection handles set patterns and filters wildcards."""

    def test_selector_set_pattern(self):
        """Selector with '{Dr,R2}|*' protects all matching genotypes."""
        sp = nt.Species.from_dict(
            "shc_setpat", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = nt.HomingDrive(
            name="d", drive_allele="Dr", target_allele="WT",
            resistance_allele="R2", drive_conversion_rate=1.0,
        )

        @nt.hook(event="first", priority=0,
                 selectors={"drive_carriers": "{Dr,R2}|*"})
        def count_drive(state, config, drive_carriers, deme_id=-1):
            _ = (state, config, drive_carriers, deme_id)

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).hooks(count_drive).build()

        reg = pop.demes[0].index_registry
        # Dr|WT should survive — matched by {Dr,R2}|*
        dr_wt = sp.get_genotype_from_str("Dr|WT")
        assert reg.ztype_index(dr_wt, "default") >= 0

    def test_selector_wildcard_only_is_filtered(self):
        """Selector with '*' only — no concrete genotypes extracted."""
        sp = nt.Species.from_dict(
            "shc_selwild", {"c1": {"l1": ["WT", "Dr"]}},
            unordered=True, gamete_labels=["default"],
        )

        @nt.hook(event="first", priority=0, selectors={"all": "*"})
        def count_all(state, config, all, deme_id=-1):
            _ = (state, config, all, deme_id)

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).hooks(count_all).build()

        reg = pop.demes[0].index_registry
        # Only genotypes reachable from initial state should survive
        # (the wildcard * does NOT add seeds)
        wt_wt = sp.get_genotype_from_str("WT|WT")
        assert reg.ztype_index(wt_wt, "default") >= 0


class TestSpatialCustomHookSkipped:
    """Custom hooks (custom=True) are explicitly NOT auto-collected."""

    def test_custom_hook_not_collected(self):
        """Custom hook referencing Dr|Dr — genotype is pruned."""
        sp = nt.Species.from_dict(
            "shc_custom", {"c1": {"l1": ["WT", "Dr"]}},
            unordered=True, gamete_labels=["default"],
        )

        @nt.hook(event="first", custom=True)
        def custom_release(state, config, deme_id=-1):
            # Reference Dr|Dr only in the body — not detectable statically
            _ = (state, config, deme_id)

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(stochastic=False, compress=True).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).hooks(custom_release).build()

        reg = pop.demes[0].index_registry
        # Dr|Dr should NOT be in the registry — custom hooks aren't collected
        dr_dr = sp.get_genotype_from_str("Dr|Dr")
        with pytest.raises((ValueError, LookupError, KeyError)):
            reg.ztype_index(dr_dr, "default")


class TestSpatialDeclaredZygoteTypes:
    """Manual declared_zygote_types supports pattern formats and fuzzy matching."""

    def test_declared_wildcard_pattern(self):
        """declared_zygote_types with 'Dr|*' protects all Dr genotypes."""
        sp = nt.Species.from_dict(
            "shc_decl_wild", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = nt.HomingDrive(
            name="d", drive_allele="Dr", target_allele="WT",
            resistance_allele="R2", drive_conversion_rate=1.0,
        )

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(
            stochastic=False, compress=True, declared_zygote_types={"Dr|*"},
        ).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).build()

        reg = pop.demes[0].index_registry
        dr_dr = sp.get_genotype_from_str("Dr|Dr")
        dr_wt = sp.get_genotype_from_str("Dr|WT")
        assert reg.ztype_index(dr_dr, "default") >= 0
        assert reg.ztype_index(dr_wt, "default") >= 0

    def test_declared_set_pattern(self):
        """declared_zygote_types with '{Dr,R2}::*' unordered set."""
        sp = nt.Species.from_dict(
            "shc_decl_set", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = nt.HomingDrive(
            name="d", drive_allele="Dr", target_allele="WT",
            resistance_allele="R2", drive_conversion_rate=1.0,
        )

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(
            stochastic=False, compress=True,
            declared_zygote_types={"{Dr,R2}::*"},
        ).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).build()

        reg = pop.demes[0].index_registry
        dr_dr = sp.get_genotype_from_str("Dr|Dr")
        assert reg.ztype_index(dr_dr, "default") >= 0

    def test_declared_mixed_exact_and_pattern(self):
        """declared_zygote_types mixes exact string and pattern."""
        sp = nt.Species.from_dict(
            "shc_decl_mix", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = nt.HomingDrive(
            name="d", drive_allele="Dr", target_allele="WT",
            resistance_allele="R2", drive_conversion_rate=1.0,
        )

        pop = nt.SpatialPopulation.builder(
            species=sp, n_demes=1, pop_type="discrete_generation",
        ).setup(
            stochastic=False, compress=True,
            declared_zygote_types={"WT|WT", "Dr|*"},
        ).initial_state(
            individual_count={"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}},
        ).competition(
            carrying_capacity=10000, juvenile_growth_mode=nt.NO_COMPETITION,
        ).presets(drive).build()

        reg = pop.demes[0].index_registry
        dr_dr = sp.get_genotype_from_str("Dr|Dr")
        assert reg.ztype_index(dr_dr, "default") >= 0


class TestNonSpatialHookCompress:
    """Panmictic (non-spatial) path also auto-collects hook genotypes."""

    def test_age_structured_hook_survives_compression(self):
        """AgeStructuredPopulation.setup(compress=True).hooks() protects hook refs."""
        sp = nt.Species.from_dict(
            "nsc_hook", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            unordered=True, gamete_labels=["default"],
        )
        drive = nt.HomingDrive(
            name="d", drive_allele="Dr", target_allele="WT",
            resistance_allele="R2", drive_conversion_rate=1.0,
        )

        @nt.hook(event="first", priority=0)
        def release():
            return [nt.Op.add(genotypes="Dr|Dr", ages=1, sex="male", delta=10)]

        pop = (
            nt.AgeStructuredPopulation.setup(
                species=sp, stochastic=False, compress=True,
            )
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                {"female": {"WT|WT": [0, 100, 0]}, "male": {"WT|WT": [0, 100, 0]}}
            )
            .competition(carrying_capacity=1000, low_density_growth_rate=1)
            .presets(drive)
            .hooks(release)
            .build()
        )

        reg = pop.registry
        dr_dr = sp.get_genotype_from_str("Dr|Dr")
        # Must survive compression even though initial state only has WT|WT
        assert reg.ztype_index(dr_dr, "default") >= 0
