"""Parental deposition edits embryos independently of inherited drive/Cas9.

HomingDrive uses only deposited gamete labels to trigger embryo resistance.
For a single active source with rate r, each WT copy converts independently:
WT/WT yields ((1-r)^2, 2*r*(1-r), r^2), while Dr/WT yields (1-r, r).
The explicitly enabled paternal source acts on the maternal remainder.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.model import build_discrete_engine_config


@pytest.fixture
def host() -> Iterator[SimpleNamespace]:
    """A minimal RecipeHost over a single-locus species with a deposition glab."""
    species = nt.Species.from_dict(
        name="_deposition_carryover",
        structure={"chr1": {"A": ["WT", "Dr", "R2"]}},
        gamete_labels=["default", "cas9_deposited"],
    )
    registry = build_registry(species)
    z2g, g2z = project_mendelian_maps(species, registry)
    config = build_discrete_engine_config(
        n_genotypes=len(registry.index_to_genotype),
        n_gtypes=len(registry.index_to_gtype),
        n_glabs=len(registry.glab_labels),
        n_slabs=len(registry.slab_labels),
        zygotes_to_gametes_map=z2g,
        gametes_to_zygotes_map=g2z,
        has_sex_chromosomes=False,
    )
    yield SimpleNamespace(
        species=species, registry=registry, index_registry=registry, config=config
    )


def _row_named(host: SimpleNamespace, row: dict[int, float] | None) -> dict[str, float]:
    """Render one zygote branch row as {genotype@slab: probability}."""
    named: dict[str, float] = {}
    for zidx, prob in (row or {}).items():
        gt, slab = host.registry.index_to_ztype[zidx]
        named[f"{gt.to_string()}@{slab}"] = pytest.approx(prob)
    return named


def _deposited_wt_pair(host: SimpleNamespace) -> tuple[int, int]:
    """(maternal tagged WT, paternal default WT) gamete compressed indices."""
    reg = host.registry
    maternal = reg.gtype_index(
        host.species.get_haploid_genotype_from_str("WT"), "cas9_deposited"
    )
    paternal = reg.gtype_index(
        host.species.get_haploid_genotype_from_str("WT"), "default"
    )
    return (maternal, paternal)


def test_homing_maternal_deposition_edits_non_drive_embryo(host: SimpleNamespace) -> None:
    """A deposited-Cas9 WT|WT embryo gets embryo resistance at the embryo rate."""
    drive = nt.HomingDrive(
        name="h_carryover",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        embryo_resistance_formation_rate=0.5,
        cas9_deposition_glab="cas9_deposited",
    )
    drive.bind_species(host.species)
    rows = drive.zygote_modifier(host)()
    row = rows.get(_deposited_wt_pair(host))
    # Per-copy independent conversion at r=0.5 on the WT|WT embryo.
    assert _row_named(host, row) == {
        "WT|WT@default": pytest.approx(0.25),
        "WT|R2@default": pytest.approx(0.5),
        "R2|R2@default": pytest.approx(0.25),
    }


def test_homing_maternal_deposition_edits_drive_embryo(host: SimpleNamespace) -> None:
    """Inherited drive adds no editing beyond the maternal deposition rate."""
    drive = nt.HomingDrive(
        name="h_carryover_inherited",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        embryo_resistance_formation_rate=0.5,
        cas9_deposition_glab="cas9_deposited",
    )
    drive.bind_species(host.species)
    rows = drive.zygote_modifier(host)()
    reg = host.registry
    # Dr gamete from the carrier mother is also tagged; embryo is Dr|WT.
    pair = (
        reg.gtype_index(host.species.get_haploid_genotype_from_str("Dr"), "cas9_deposited"),
        reg.gtype_index(host.species.get_haploid_genotype_from_str("WT"), "default"),
    )
    row = rows.get(pair)
    assert _row_named(host, row) == {
        "WT|Dr@default": pytest.approx(0.5),
        "Dr|R2@default": pytest.approx(0.5),
    }


def test_toxin_antidote_maternal_deposition_disrupts_non_drive_embryo(host: SimpleNamespace) -> None:
    """A deposited-Cas9 WT|WT embryo gets embryo disruption at the embryo rate."""
    preset = nt.ToxinAntidoteDrive(
        name="ta_carryover",
        drive_allele="Dr",
        target_allele="WT",
        disrupted_allele="R2",
        embryo_disruption_rate=0.5,
        cas9_deposition_glab="cas9_deposited",
    )
    preset.bind_species(host.species)
    rows = preset.zygote_modifier(host)()
    row = rows.get(_deposited_wt_pair(host))
    # Per-copy independent disruption at r=0.5 on the WT|WT embryo.
    assert _row_named(host, row) == {
        "WT|WT@default": pytest.approx(0.25),
        "WT|R2@default": pytest.approx(0.5),
        "R2|R2@default": pytest.approx(0.25),
    }


@pytest.mark.parametrize("paternal_enabled", [False, True])
def test_homing_without_deposition_label_has_no_embryo_modifier(
    host: SimpleNamespace, paternal_enabled: bool,
) -> None:
    """Nonzero embryo rates never fall back to the embryo's own Cas9."""
    drive = nt.HomingDrive(
        name="no_deposition", drive_allele="Dr", target_allele="WT",
        resistance_allele="R2", embryo_resistance_formation_rate=0.3,
        use_paternal_deposition=paternal_enabled,
    )
    drive.bind_species(host.species)
    assert drive.zygote_modifier(host) is None


@pytest.mark.parametrize("maternal_tagged", [False, True])
@pytest.mark.parametrize("paternal_tagged", [False, True])
@pytest.mark.parametrize("paternal_enabled", [False, True])
def test_homing_edits_only_from_enabled_deposition_sources(
    host: SimpleNamespace, maternal_tagged: bool, paternal_tagged: bool,
    paternal_enabled: bool,
) -> None:
    """Dr/WT is edited according to source labels, never inherited drive."""
    drive = nt.HomingDrive(
        name="source_matrix", drive_allele="Dr", target_allele="WT",
        resistance_allele="R2",
        embryo_resistance_formation_rate={"female": 0.2, "male": 0.4},
        cas9_deposition_glab="cas9_deposited",
        use_paternal_deposition=paternal_enabled,
    )
    drive.bind_species(host.species)
    modifier = drive.zygote_modifier(host)
    assert modifier is not None
    pair = tuple(
        host.registry.gtype_index(
            host.species.get_haploid_genotype_from_str(allele),
            "cas9_deposited" if tagged else "default",
        )
        for allele, tagged in [("Dr", maternal_tagged), ("WT", paternal_tagged)]
    )
    row = modifier().get(pair)
    remaining = (0.8 if maternal_tagged else 1.0) * (
        0.6 if paternal_tagged and paternal_enabled else 1.0
    )
    if remaining == 1.0:
        # Conversion modifiers are sparse; no row means Mendelian identity.
        assert row is None or _row_named(host, row) == {"WT|Dr@default": pytest.approx(1.0)}
    else:
        assert _row_named(host, row) == {
            "WT|Dr@default": pytest.approx(remaining),
            "Dr|R2@default": pytest.approx(1.0 - remaining),
        }


@pytest.mark.parametrize("paternal_enabled", [False, True])
def test_homing_zero_active_rates_produce_no_modifier(
    host: SimpleNamespace, paternal_enabled: bool,
) -> None:
    """A disabled paternal source cannot activate embryo editing."""
    drive = nt.HomingDrive(
        name="inactive_rates", drive_allele="Dr", target_allele="WT",
        embryo_resistance_formation_rate=(0.0, 0.0 if paternal_enabled else 0.5),
        cas9_deposition_glab="cas9_deposited",
        use_paternal_deposition=paternal_enabled,
    )
    drive.bind_species(host.species)
    assert drive.zygote_modifier(host) is None


@pytest.mark.parametrize("embryo_rate,functional_ratio", [(0.3, 0.25), (1.0, 1.0)])
def test_homing_deposition_functional_resistance_split(
    embryo_rate: float, functional_ratio: float,
) -> None:
    """Deposited Cas9 preserves the requested absolute R1/R2 proportions."""
    species = nt.Species.from_dict(
        name="functional_deposition",
        structure={"chr1": {"A": ["WT", "Dr", "R1", "R2"]}},
        gamete_labels=["default", "cas9_deposited"],
    )
    drive = nt.HomingDrive(
        name="functional", drive_allele="Dr", target_allele="WT",
        resistance_allele="R2", functional_resistance_allele="R1",
        drive_conversion_rate=0, embryo_resistance_formation_rate=embryo_rate,
        functional_resistance_ratio=functional_ratio,
        cas9_deposition_glab="cas9_deposited",
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
        .initial_state({"female": {"Dr|Dr": 100}, "male": {"WT|WT": 100}})
        .presets(drive).build()
    )
    registry = pop.registry
    pair = (
        registry.gtype_index(species.get_haploid_genotype_from_str("Dr"), "cas9_deposited"),
        registry.gtype_index(species.get_haploid_genotype_from_str("WT"), "default"),
    )
    expected = {"WT": 1 - embryo_rate, "R1": embryo_rate * functional_ratio,
                "R2": embryo_rate * (1 - functional_ratio)}
    actual = {}
    for zidx, (genotype, _slab) in enumerate(registry.index_to_ztype):
        probability = pop.config.gametes_to_zygotes_map[pair[0], pair[1], zidx]
        if probability:
            actual[genotype.to_string()] = probability
    assert actual == {
        species.get_genotype_from_str(f"Dr|{allele}").to_string(): pytest.approx(value, abs=1e-14)
        for allele, value in expected.items() if value
    }


@pytest.mark.parametrize("mother", ["WT|Dr;n|C", "WT|Dr;n|n", "WT|WT;n|C"])
def test_split_drive_deposition_requires_both_parental_components(mother: str) -> None:
    """Only a mother with drive and Cas9 deposits; noninheriting embryos edit."""
    species = nt.Species.from_dict(
        name="split_deposition",
        structure={"chr1": {"A": ["WT", "Dr", "R2"]}, "chr2": {"B": ["n", "C"]}},
        gamete_labels=["default", "cas9_deposited"],
    )
    drive = nt.HomingDrive(
        name="split", drive_allele="Dr", cas9_allele="C", target_allele="WT",
        resistance_allele="R2", drive_conversion_rate=0,
        embryo_resistance_formation_rate=1,
        cas9_deposition_glab="cas9_deposited",
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False, compress=False)
        .initial_state({"female": {mother: 100}, "male": {"WT|WT;n|n": 100}})
        .presets(drive).build()
    )
    registry = pop.registry
    mother_idx = registry.ztype_index(species.get_genotype_from_str(mother), "default")
    maternal_row = pop.config.zygotes_to_gametes_map[0, mother_idx]
    deposits = mother == "WT|Dr;n|C"
    deposited_mass = sum(
        maternal_row[i] for i, (_genotype, label) in enumerate(registry.index_to_gtype)
        if label == "cas9_deposited"
    )
    assert deposited_mass == pytest.approx(float(deposits), abs=1e-14)
    # Every mother can emit WT;n. Its embryo inherits neither drive nor Cas9,
    # yet a deposit converts both WT copies with rate one.
    maternal_idx = registry.gtype_index(
        species.get_haploid_genotype_from_str("WT;n"),
        "cas9_deposited" if deposits else "default",
    )
    assert maternal_row[maternal_idx] > 0
    paternal_idx = registry.gtype_index(species.get_haploid_genotype_from_str("WT;n"), "default")
    expected_genotype = "R2|R2;n|n" if deposits else "WT|WT;n|n"
    expected_idx = registry.ztype_index(species.get_genotype_from_str(expected_genotype), "default")
    row = pop.config.gametes_to_zygotes_map[maternal_idx, paternal_idx]
    assert row[expected_idx] == pytest.approx(1.0, abs=1e-14)
    assert row.sum() == pytest.approx(1.0, abs=1e-14)
