"""R3-05: migration mixing dynamics, Fisherian sex ratio, Wolbachia contract.

Requirements under attack:

1. Two-deme migration with per-deme emigration proportion m (documented:
   "Per-deme proportion of individuals migrating per step").  With no
   selection and no growth difference, the allele-frequency difference
   between the demes decays geometrically,
       Delta(t) = Delta(0) * (1 - 2 m)^t,
   and the mass-weighted global allele frequency is invariant.  This is
   the standard two-deme mixing result and is independent of the deme
   sizes, the kernel normalization, and whether migration runs before or
   after reproduction (the migrant pool inherits its source deme's
   composition).

2. Fisher's sex-ratio argument / chromosomal sex determination: in an XY
   species every daughter inherits the paternal X and every son the
   paternal Y, so the offspring sex ratio is exactly 1:1 -- starting from
   any parental sex ratio and independently of the `sex_ratio`
   parameter.  An unbalanced population must return to 1:1 in one
   generation.

3. Wolbachia contract (docs/en/4_index_registry.md: "slabs are used by
   concrete Presets such as **Wolbachia** (cytoplasmic incompatibility
   modelled with the default infected/normal slabs and a wolbachia
   gamete label)").  Cytoplasmic incompatibility means an infected male
   crossed with an uninfected female has reduced or zero offspring,
   while the reciprocal cross is fertile; a strain with perfect maternal
   transmission and incompatibility spreads to fixation above the
   Caspari-Watson threshold.  FINDING F3 (marked EVIDENCE below): the
   preset implements only maternal slab inheritance plus per-slab
   fitness scaling -- there is no incompatibility cross effect and no
   transmission-fidelity parameter.  The documented CI is absent from
   the implementation.

Wrong results rejected: migration that mixes without any decay law or
loses mass-weighted frequency, migration that changes the global
frequency, an XY species whose offspring sex ratio follows the
`sex_ratio` parameter instead of the chromosomes, and a documented CI
capability that silently does nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import natal as nt
from natal.frontend.presets.cytoplasmic import Wolbachia
from natal.frontend.spatial.builder import batch_setting

TOL = 1e-9


# --------------------------------------------------------------------------
# migration mixing
# --------------------------------------------------------------------------
def _two_deme_pop(name: str, *, migration_rate: float, deme1: dict, deme2: dict):
    """Two fully connected demes.

    The adjacency matrix is passed explicitly: a migration rate without a
    topology, kernel, or adjacency silently migrates nobody (the docs'
    examples always supply a topology such as ``HexGrid``).
    """
    species = nt.Species.from_dict(
        name=f"{name}_sp",
        structure={"chr1": {"loc": ["W", "D"]}},
        gamete_labels=["default"],
    )
    counts = batch_setting([deme1, deme2])
    builder = nt.SpatialPopulation.builder(
        species, n_demes=2, pop_type="discrete_generation"
    )
    return (
        builder.setup(name=name, stochastic=False)
        .initial_state(individual_count=counts)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .competition(juvenile_growth_mode="no_competition")
        .migration(
            adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), migration_rate=migration_rate
        )
        .build()
    )


def _deme_allele_frequency(pop, deme_index: int, allele: str) -> float:
    deme = pop.demes[deme_index]
    counts = np.asarray(deme.state.individual_count)
    total = float(counts.sum())
    copies = 0.0
    for genotype, slab in deme.index_registry.index_to_ztype:
        idx = deme.index_registry.ztype_index(genotype, slab)
        tokens = [
            token
            for chunk in genotype.to_string().split(";")
            for token in chunk.split("|")
        ]
        copies += float(tokens.count(allele)) * float(counts[:, :, idx].sum())
    return copies / (2.0 * total) if total > 0 else float("nan")


def _deme_totals(pop) -> np.ndarray:
    return np.array([float(d.state.individual_count.sum()) for d in pop.demes])


def test_two_deme_frequency_difference_decays_geometrically() -> None:
    """m = 0.2: Delta(t) = 0.6^t and the global frequency is invariant."""
    m = 0.2
    pop = _two_deme_pop(
        "R3_05_mix",
        migration_rate=m,
        deme1={"female": {"W|W": 500.0}, "male": {"W|W": 500.0}},
        deme2={"female": {"D|D": 500.0}, "male": {"D|D": 500.0}},
    )
    q1 = _deme_allele_frequency(pop, 0, "D")
    q2 = _deme_allele_frequency(pop, 1, "D")
    assert q1 == pytest.approx(0.0, abs=TOL)
    assert q2 == pytest.approx(1.0, abs=TOL)
    delta0 = q1 - q2

    for tick in range(1, 9):
        pop.run(1)
        q1 = _deme_allele_frequency(pop, 0, "D")
        q2 = _deme_allele_frequency(pop, 1, "D")
        delta = q1 - q2
        assert delta == pytest.approx(delta0 * (1.0 - 2.0 * m) ** tick, abs=1e-9), tick
        # Mass-weighted global frequency is conserved exactly (no selection,
        # equal deme totals under symmetric exchange).
        totals = _deme_totals(pop)
        assert totals[0] == pytest.approx(totals[1], rel=1e-9)
        global_q = (q1 * totals[0] + q2 * totals[1]) / (totals[0] + totals[1])
        assert global_q == pytest.approx(0.5, abs=1e-9), tick


def test_migration_rate_zero_leaves_demes_untouched() -> None:
    """m = 0: the frequency difference does not decay at all."""
    pop = _two_deme_pop(
        "R3_05_nomix",
        migration_rate=0.0,
        deme1={"female": {"W|W": 500.0}, "male": {"W|W": 500.0}},
        deme2={"female": {"D|D": 500.0}, "male": {"D|D": 500.0}},
    )
    for _ in range(5):
        pop.run(1)
    assert _deme_allele_frequency(pop, 0, "D") == pytest.approx(0.0, abs=TOL)
    assert _deme_allele_frequency(pop, 1, "D") == pytest.approx(1.0, abs=TOL)


# --------------------------------------------------------------------------
# Fisherian sex ratio under XY
# --------------------------------------------------------------------------
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


@pytest.mark.parametrize("sex_ratio", [0.5, 0.9])
def test_xy_offspring_sex_ratio_is_exactly_one_to_one(sex_ratio: float) -> None:
    """100 females and 1000 males produce exactly 200 daughters and 200 sons."""
    species = _xy_species(f"R3_05_xy_{sex_ratio}")
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=f"R3_05_xy_{sex_ratio}", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"A|A;X1|X1": 100.0},
                "male": {"a|a;X1|Y1": 1000.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=4.0, sex_ratio=sex_ratio)
        .competition(juvenile_growth_mode="no_competition", carrying_capacity=1e12)
        .build()
    )
    population.run(1)
    counts = np.asarray(population.state.individual_count)
    females = float(counts[0, 1, :].sum())
    males = float(counts[1, 1, :].sum())
    assert females == pytest.approx(200.0, rel=1e-12)
    assert males == pytest.approx(200.0, rel=1e-12)
    assert females + males == pytest.approx(400.0, rel=1e-12)
    # Daughters are XX, sons are XY: genetic sex is not a ratio knob.
    labels = {
        index: genotype.to_string()
        for index, (genotype, _) in enumerate(population.registry.index_to_ztype)
    }
    for index, label in labels.items():
        if "Y" in label:
            assert counts[0, 1, index] == 0.0, label
        else:
            assert counts[1, 1, index] == 0.0, label


# --------------------------------------------------------------------------
# Wolbachia contract
# --------------------------------------------------------------------------
def _wol_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT"]}},
        gamete_labels=["default", "wolbachia"],
        somatic_labels=["normal", "infected"],
    )


def _slab_totals(pop):
    normal = infected = 0.0
    for genotype, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype, slab)
        mass = float(np.asarray(pop.state.individual_count[:, 1, idx]).sum())
        if slab == "infected":
            infected += mass
        else:
            normal += mass
    return normal, infected


def _wol_cross(name: str, *, female_key: str, male_key: str, viability: float = 1.0):
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_wol_species(f"{name}_sp"), name=name, stochastic=False
        )
        .presets(Wolbachia(name=f"{name}_wol", viability_scaling=viability))
        .initial_state(
            individual_count={"female": {female_key: 500.0}, "male": {male_key: 500.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )
    population.run(1)
    return _slab_totals(population)


def test_docs_and_preset_agree_that_ci_is_not_implemented() -> None:
    """The Wolbachia contract now matches the preset.

    The index-registry docs used to claim the preset models cytoplasmic
    incompatibility while the preset only implements maternal slab inheritance
    plus per-slab fitness scaling; the docs were corrected (both languages).
    This guard fails if the false claim returns, and the companion test below
    pins what the preset actually does.
    """
    root = Path(__file__).resolve().parents[2]
    for relative in ("docs/en/4_index_registry.md", "docs/zh/4_index_registry.md"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "cytoplasmic incompatibility modelled with the default" not in text, relative
        assert "建模细胞质不兼容性，并要求" not in text, relative
        marker = "does not implement cytoplasmic incompatibility" if "en/" in relative else "不实现细胞质不兼容"
        assert marker in text, relative


def test_implemented_wolbachia_is_a_neutral_maternal_marker() -> None:
    """What the preset does implement: exact maternal slab inheritance.

    With no fitness cost the infection frequency is invariant (each
    infected mother transmits to every offspring, no paternal effect), and
    an infected father transmits nothing.  A CI-bearing strain would
    instead rise to fixation from above its threshold.
    """
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_wol_species("R3_05_wol_neutral_sp"),
            name="R3_05_wol_neutral",
            stochastic=False,
        )
        .presets(Wolbachia(name="R3_05_wol_neutral_preset", viability_scaling=1.0))
        .initial_state(
            individual_count={
                "female": {"WT|WT@infected": 500.0, "WT|WT": 500.0},
                "male": {"WT|WT@infected": 500.0, "WT|WT": 500.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )
    for _ in range(5):
        population.run(1)
        normal, infected = _slab_totals(population)
        # Infection frequency is pinned at the maternal frequency (0.5).
        assert infected / (infected + normal) == pytest.approx(0.5, abs=1e-9)

    # Infected fathers contribute nothing to the offspring slab.
    normal, infected = _wol_cross("R3_05_wol_paternal", female_key="WT|WT", male_key="WT|WT@infected")
    assert infected == pytest.approx(0.0, abs=TOL)
    assert normal == pytest.approx(1000.0, abs=TOL)
