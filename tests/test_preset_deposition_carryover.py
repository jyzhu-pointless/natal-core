"""Deposition carryover contract for the drive presets (evaluator regression).

Confirmed requirement: ``cas9_deposition_glab`` models the maternal/paternal
Cas9 deposition effect — embryos formed from *deposited* gametes undergo
embryo editing even when they did not inherit the drive allele.  The preset
docstrings state this directly:

- HomingDrive.zygote_modifier: "Cleavage in the embryo (due to deposited
  Cas9 or zygotic expression) converts wild-type alleles into resistance
  alleles."
- ToxinAntidoteDrive's deposition glab: "The zygote modifier will read
  this tag to apply embryo resistance." (gamete_modifier comment)

Concretely, for a WT gamete tagged ``cas9_deposited`` (produced by a
drive-carrier mother) fertilizing a default WT gamete, the resulting
``WT|WT`` embryo must be edited at the configured embryo rate with
independent per-copy conversion — historically
``{WT|WT: (1-r)^2, one-copy: 2r(1-r), two-copy: r^2}`` for rate ``r``.

CR-1 regression check (2026-09-13): during the conversion-rules migration
the zygote embryo rules gained an extra ``filters={"current": carrier}``
restriction, which silently disables this deposition-carryover pathway:
the WT|WT embryo from a deposited gamete is no longer edited at all.
Executed evidence at review time:

- old (HEAD 084c8f4): ``{'WT|WT@default': 0.25, 'WT|R2@default': 0.5,
  'R2|R2@default': 0.25}``
- new (working tree): ``{'WT|WT@default': 1.0}``

These tests encode the documented (old) contract and are the repair
targets for the migration.
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


def test_homing_maternal_deposition_edits_non_drive_embryo(host) -> None:
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


def test_homing_maternal_deposition_edits_drive_embryo(host) -> None:
    """The inherited-drive pathway keeps working under the deposition glab.

    Pinned to the executed pre-migration behavior: the female (deposition)
    rule and the male (zygotic-expression) rule both fire in cascade on a
    drive-carrying embryo, so the remaining WT copy sees the rate twice.
    Old and new code agree here; this guards against repair overreach.
    """
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
        "WT|Dr@default": pytest.approx(0.25),
        "Dr|R2@default": pytest.approx(0.75),
    }


def test_toxin_antidote_maternal_deposition_disrupts_non_drive_embryo(host) -> None:
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
