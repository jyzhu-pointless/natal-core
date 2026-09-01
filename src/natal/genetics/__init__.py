"""Forwarding shim: the ``genetics`` package now lives at
``natal.frontend.genetics``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.genetics.entities as _m0
import natal.frontend.genetics.entities._base as _m1
import natal.frontend.genetics.entities.gene as _m2
import natal.frontend.genetics.entities.genotype as _m3
import natal.frontend.genetics.entities.haplotype as _m4
import natal.frontend.genetics.structures as _m5
import natal.frontend.genetics.structures._base as _m6
import natal.frontend.genetics.structures._construction as _m7
import natal.frontend.genetics.structures._enumeration as _m8
import natal.frontend.genetics.structures._helpers as _m9
import natal.frontend.genetics.structures._mapping as _m10
import natal.frontend.genetics.structures._pattern as _m11
import natal.frontend.genetics.structures._registry as _m12
import natal.frontend.genetics.structures._types as _m13
import natal.frontend.genetics.structures.chromosome as _m14
import natal.frontend.genetics.structures.chromosome_map as _m15
import natal.frontend.genetics.structures.locus as _m16
import natal.frontend.genetics.structures.species as _m17
from natal.frontend.genetics import (
    Allele,
    Chromosome,
    DiploidGenome,
    DiploidGenotype,
    Gene,
    Genome,
    GenomeTemplate,
    Genotype,
    HaploidGenome,
    HaploidGenotype,
    Haplotype,
    Karyotype,
    Linkage,
    Locus,
    RecombinationMap,
    SexChromosomeType,
    Species,
    SpeciesConfigBlueprint,
    build_compression_mask,
    compute_recombinant_haplotypes,
    compute_recombinant_haplotypes_with_alleles,
    create_chromosome_from_allele_names,
    create_haplotype_from_allele_names,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.genetics.entities"] = _m0
_sys.modules["natal.genetics.entities._base"] = _m1
_sys.modules["natal.genetics.entities.gene"] = _m2
_sys.modules["natal.genetics.entities.genotype"] = _m3
_sys.modules["natal.genetics.entities.haplotype"] = _m4
_sys.modules["natal.genetics.structures"] = _m5
_sys.modules["natal.genetics.structures._base"] = _m6
_sys.modules["natal.genetics.structures._construction"] = _m7
_sys.modules["natal.genetics.structures._enumeration"] = _m8
_sys.modules["natal.genetics.structures._helpers"] = _m9
_sys.modules["natal.genetics.structures._mapping"] = _m10
_sys.modules["natal.genetics.structures._pattern"] = _m11
_sys.modules["natal.genetics.structures._registry"] = _m12
_sys.modules["natal.genetics.structures._types"] = _m13
_sys.modules["natal.genetics.structures.chromosome"] = _m14
_sys.modules["natal.genetics.structures.chromosome_map"] = _m15
_sys.modules["natal.genetics.structures.locus"] = _m16
_sys.modules["natal.genetics.structures.species"] = _m17

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "Allele",
    "Chromosome",
    "DiploidGenome",
    "DiploidGenotype",
    "Gene",
    "Genome",
    "GenomeTemplate",
    "Genotype",
    "HaploidGenome",
    "HaploidGenotype",
    "Haplotype",
    "Karyotype",
    "Linkage",
    "Locus",
    "RecombinationMap",
    "SexChromosomeType",
    "Species",
    "SpeciesConfigBlueprint",
    "build_compression_mask",
    "compute_recombinant_haplotypes",
    "compute_recombinant_haplotypes_with_alleles",
    "create_chromosome_from_allele_names",
    "create_haplotype_from_allele_names",
]
