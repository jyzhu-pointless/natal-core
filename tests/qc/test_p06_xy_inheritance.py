"""P6: XY sex-chromosome inheritance through the public discrete path.

Claim: in an XY species the father's chromosome pair (X/Y) partitions
offspring sex -- every daughter receives the paternal X, every son the
paternal Y; the maternal X is inherited by all offspring.  Sex ratio
parameters must not move this split (genetic sex wins), and no zygote of
one genetic sex may appear in the other sex's count plane.

Reference: chromosome-level Mendelian segregation stated in the species
structure, independent of the blueprint mask implementation.

Wrong results rejected: YY genotypes, paternal X reaching sons, sex_ratio
distorting the 1:1 genetic sex split, mask leakage across sex planes.
"""

from __future__ import annotations

import natal as nt
import numpy as np

TOL = 1e-9


def _xy_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1", "X2"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )


def _build(species: nt.Species, name: str, female_key: str, male_key: str, sex_ratio: float):
    female = species.get_genotype_from_str(female_key)
    male = species.get_genotype_from_str(male_key)
    builder = nt.DiscreteGenerationPopulation.setup(
        species=species, name=name, stochastic=False
    )
    return (
        builder.initial_state(
            individual_count={"female": {female: 500}, "male": {male: 500}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=4.0, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode="no_competition",
            carrying_capacity=1e12,
            low_density_growth_rate=2.0,
        )
        .build()
    )


def test_paternal_x_y_partition_is_exact() -> None:
    species = _xy_species("QC0915_p06a")
    pop = _build(species, "QC0915_p06a", "A|A;X1|X1", "a|a;X1|Y1", 0.5)
    pop.run(1)
    counts = np.asarray(pop.state.individual_count)

    daughters = species.get_genotype_from_str("A|a;X1|X1")
    sons = species.get_genotype_from_str("A|a;X1|Y1")
    registry = pop.registry

    def index_of(genotype):
        for g, slab in registry.index_to_ztype:
            if g is genotype or g.to_string() == genotype.to_string():
                return registry.ztype_index(g, slab)
        raise KeyError(genotype.to_string())

    # 500 females x 4 eggs = 2000 zygotes: 1000 daughters, 1000 sons.
    d_idx, s_idx = index_of(daughters), index_of(sons)
    assert counts[0, 1, d_idx] == 1000.0
    assert counts[1, 1, s_idx] == 1000.0
    assert counts.sum() == 2000.0
    # No other ztype received mass, and no zygote appears across the sex
    # axis from its genetic sex (Y-bearing are male, X/X are female).
    for g, slab in registry.index_to_ztype:
        idx = registry.ztype_index(g, slab)
        s = g.to_string()
        if s in ("A|a;X1|X1", "A|a;X1|Y1"):
            continue
        assert counts[:, :, idx].sum() == 0.0, s
    assert counts[1, :, d_idx].sum() == 0.0
    assert counts[0, :, s_idx].sum() == 0.0


def test_sex_ratio_cannot_move_genetic_sex() -> None:
    species = _xy_species("QC0915_p06b")
    pop = _build(species, "QC0915_p06b", "A|A;X1|X1", "a|a;X1|Y1", 0.9)
    pop.run(1)
    counts = np.asarray(pop.state.individual_count)
    # Genetic sex wins: still exactly 1000/1000 despite sex_ratio 0.9.
    assert counts[0, :, :].sum() == 1000.0
    assert counts[1, :, :].sum() == 1000.0


def test_maternal_x_segregates_50_50_in_sons_and_daughters() -> None:
    species = _xy_species("QC0915_p06c")
    pop = _build(species, "QC0915_p06c", "A|A;X1|X2", "a|a;X1|Y1", 0.5)
    pop.run(1)
    counts = np.asarray(pop.state.individual_count)
    registry = pop.registry
    per_z = {}
    for g, slab in registry.index_to_ztype:
        idx = registry.ztype_index(g, slab)
        per_z[g.to_string()] = counts[:, :, idx].sum()
    # Mother's X1/X2 segregates 50/50; father contributes X1 (daughters)
    # or Y1 (sons).  All four X-combinations appear at 500 each.
    assert abs(per_z.get("A|a;X1|X1", 0.0) - 500.0) < TOL
    assert abs(per_z.get("A|a;X2|X1", 0.0) - 500.0) < TOL
    assert abs(per_z.get("A|a;X1|Y1", 0.0) - 500.0) < TOL
    assert abs(per_z.get("A|a;X2|Y1", 0.0) - 500.0) < TOL
    assert counts.sum() == 2000.0
