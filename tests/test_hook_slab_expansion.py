"""Integration tests: hook genotype resolution correctly expands to
ZType indices for n_slabs > 1.

The CSR kernel uses zidx to index individual_count[sex_idx, age, zidx],
whose third dimension is n_ztypes (not num_genotypes). Before the fix,
hook selectors resolved to genotype indices, which only covered the first
slab variant of each genotype. After the fix, each genotype index is
expanded to all its ZType indices (one per surviving slab).
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.hooks.entry.declarative import (
    Op,
    _resolve_genotypes,
    compile_declarative_hook,
)
from natal.hooks.entry.selector import _resolve_selector_to_array
from natal.index_registry import IndexRegistry

# ── Helpers ─────────────────────────────────────────────────────────────

def _nslab_species(somatic_labels=None):
    return nt.Species.from_dict(
        "nslab_hook_test",
        {"c1": {"l1": ["A", "a"]}},
        gamete_labels=["default"],
        somatic_labels=somatic_labels or ["default"],
    )


def _registry_for_species(sp):
    reg = IndexRegistry()
    # Register labels before genotypes so auto-cross-product works
    reg.slab_labels = sp.somatic_labels or ["default"]
    for g in sp.get_all_genotypes():
        reg.register_genotype(g)
    return reg


# ── Phase A: n_slabs=1 regression ──────────────────────────────────────

class TestPhaseA_NSlabsOneRegression:
    """Verify n_slabs=1 behavior is unchanged (backward compat)."""

    def test_star_returns_all_ztypes(self):
        sp = _nslab_species()
        reg = _registry_for_species(sp)
        result = _resolve_genotypes("*", reg, reg.index_to_genotype, reg.n_ztypes)
        expected = np.arange(reg.n_ztypes, dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_string_label_resolves_first_ztype(self):
        sp = _nslab_species()
        reg = _registry_for_species(sp)
        result = _resolve_genotypes("A|A", reg, reg.index_to_genotype, reg.n_ztypes)
        assert list(result) == [0]

    def test_list_of_strings(self):
        sp = _nslab_species()
        reg = _registry_for_species(sp)
        result = _resolve_genotypes(
            ["A|A", "a|a"], reg, reg.index_to_genotype, reg.n_ztypes,
        )
        # 3 unordered: A|A=0, A|a=1, a|a=2
        assert list(result) == [0, 2]

    def test_int_input_passthrough(self):
        sp = _nslab_species()
        reg = _registry_for_species(sp)
        result = _resolve_genotypes([0, 2], reg, reg.index_to_genotype, reg.n_ztypes)
        assert list(result) == [0, 2]

    def test_selector_star_returns_all(self):
        sp = _nslab_species()
        reg = _registry_for_species(sp)
        result = _resolve_selector_to_array("*", reg, reg.index_to_genotype)
        assert list(result) == list(range(reg.n_ztypes))


# ── Phase B: Fitness + slab + compression ───────────────────────────────

class TestPhaseB_FitnessAndCompression:
    """Hook compilation with slabs and compression."""

    def test_compile_hook_with_two_slabs(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )
        desc = compile_declarative_hook([Op.scale(factor=0.5)], pop, event="early")
        # 3 unordered genotypes × 2 slabs = 6 ZTypes
        assert len(desc.plan.zidx_data) == 6

    def test_selector_resolve_star_two_slabs(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        reg = _registry_for_species(sp)
        result = _resolve_selector_to_array("*", reg, reg.index_to_genotype)
        assert list(result) == list(range(6))

    def test_selector_resolve_specific_genotype_two_slabs(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        reg = _registry_for_species(sp)
        # A|A is genotype 0 → ZTypes 0,1
        result = _resolve_selector_to_array("A|A", reg, reg.index_to_genotype)
        assert list(result) == [0, 1]


# ── Phase C: Hook+slab n_slabs>1 (the actual bug) ──────────────────────

class TestPhaseC_HookSlabNSlabsGtOne:
    """Hooks correctly affect all slab variants when n_slabs > 1."""

    def test_star_hook_affects_all_ztypes(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@normal": 50, "A|A@exposed": 30},
                "male": {"A|A@normal": 50, "A|A@exposed": 30},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )

        @nt.hook(event="early")
        def scale_all():
            return [Op.scale(factor=0.5)]

        desc = scale_all.register(pop)
        # Verify the compiled plan targets all 6 ztypes (not just 1 slab)
        assert set(desc.plan.zidx_data.tolist()) == set(range(6))
        pop.run(1)
        assert pop.state.individual_count.sum() > 0

    def test_specific_genotype_hook_affects_all_slabs(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@normal": 50, "A|A@exposed": 30, "a|a@normal": 20},
                "male": {"A|A@normal": 50, "A|A@exposed": 30, "a|a@normal": 20},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )

        @nt.hook(event="early")
        def kill_AA():
            return [Op.kill(prob=1.0, genotypes="A|A")]

        kill_AA.register(pop)
        pop.run(1)
        ic = pop.state.individual_count
        # Both A|A slab variants (ztypes 0,1) should be killed
        assert ic[:, :, 0].sum() == 0.0
        assert ic[:, :, 1].sum() == 0.0
        # a|a (ztype 4) should survive
        assert ic[:, :, 4].sum() > 0

    def test_add_hook_affects_correct_ztypes(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@normal": 50}, "male": {"A|A@normal": 50},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )

        @nt.hook(event="early")
        def add_aA():
            return [Op.add(delta=10, genotypes="a|A")]

        add_aA.register(pop)
        pop.run(1)
        ic = pop.state.individual_count
        # a|A is genotype 1 (unordered A|a) → ztypes 2,3
        assert ic[:, :, 2].sum() > 0
        assert ic[:, :, 3].sum() > 0

    def test_sample_hook_affects_all_ztypes(self):
        sp = _nslab_species(somatic_labels=["normal", "exposed"])
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@normal": 50, "A|A@exposed": 30},
                "male": {"A|A@normal": 50, "A|A@exposed": 30},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )

        @nt.hook(event="early")
        def sample_hook():
            return [Op.sample(size=10, genotypes="A|A")]

        desc = sample_hook.register(pop)
        # Verify the compiled plan targets both slab variants of A|A
        assert set(desc.plan.zidx_data.tolist()) == {0, 1}
        pop.run(1)
        assert pop.state.individual_count.sum() > 0


# ── Phase D: Multi-locus n_slabs>1 ──────────────────────────────────────

class TestPhaseD_MultiLocusSlab:
    """Multi-locus scenarios with n_slabs > 1."""

    def test_two_locus_star_resolves_correct_count(self):
        sp = nt.Species.from_dict(
            "phD_2locus",
            {"c1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
            somatic_labels=["normal", "exposed"],
        )
        reg = _registry_for_species(sp)
        result = _resolve_genotypes("*", reg, reg.index_to_genotype, reg.n_ztypes)
        assert len(result) == reg.n_ztypes
        # 9 unordered genotypes (alleles-at-locus only) × 2 slabs = 18
        assert reg.n_ztypes == 18

    def test_two_locus_specific_genotype_all_slabs(self):
        sp = nt.Species.from_dict(
            "phD_spec",
            {"c1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
            somatic_labels=["normal", "exposed"],
        )
        reg = _registry_for_species(sp)
        result = _resolve_genotypes(
            "A1/B1|A1/B1", reg, reg.index_to_genotype, reg.n_ztypes,
        )
        assert len(result) == 2  # 2 slabs

    def test_two_locus_with_real_hook_runs(self):
        sp = nt.Species.from_dict(
            "phD_run",
            {"c1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A1/B1|A1/B1": {1: 50}},
                "male": {"A1/B1|A1/B1": {1: 50}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )

        @nt.hook(event="early")
        def scale_AA_BB():
            return [Op.scale(factor=0.5, genotypes="A1/B1|A1/B1")]

        scale_AA_BB.register(pop)
        pop.run(1)
        assert pop.state.individual_count.sum() > 0
