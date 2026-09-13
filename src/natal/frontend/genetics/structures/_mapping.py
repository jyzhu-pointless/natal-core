"""SpeciesMappingMixin — genotype/gamete mapping methods for Species."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    cast,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from natal.frontend.genetics.entities.genotype import Genotype
    from natal.frontend.genetics.entities.haplotype import HaploidGenotype

    from .species import Species, SpeciesConfigBlueprint
else:
    Species = object  # runtime stand-in for cast()


class SpeciesMappingMixin:
    """Mapping methods for Species — genotype ordering, gamete/zygote maps, config blueprint.

    Provides unordered genotype canonicalization and methods to build the
    genotype-to-gamete and gamete-to-zygote transition maps used by the
    simulation engine.
    """

    def unordered_genotype(
        self,
        hg1: HaploidGenotype,
        hg2: HaploidGenotype,
    ) -> Genotype:
        """Return a canonical Genotype where maternal/paternal order is irrelevant.

        Canonicalizes per-locus: at each locus the maternal allele has the
        smaller :meth:`Locus.allele_index`.  When individual alleles must be
        swapped between the two haploid genomes (multi-locus free combination)
        new :class:`HaploidGenotype` objects are assembled so that every
        genotype with the same per-locus allele composition collapses to the
        same canonical form.
        """
        self = cast(Species, self)
        from natal.frontend.genetics.entities.genotype import Genotype

        from ._helpers import canonical_haploid_pair
        mat, pat = canonical_haploid_pair(self, hg1, hg2)
        return Genotype(species=self, maternal=mat, paternal=pat)

    def build_gamete_map(
        self,
        gamete_modifiers: Optional[list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
        n_slabs: int = 1,
    ) -> NDArray[np.float64]:
        """Build the genotype → gamete map for this species.

        When *gamete_modifiers* is None, returns the Mendelian baseline.

        Args:
            gamete_modifiers: Optional modifier callables to apply.
            n_slabs: Number of somatic slabs.  When > 1 the genotype axis is
                tiled so that each base genotype appears once per slab.
        """
        self = cast(Species, self)
        from natal.frontend.genetics import initialize_gamete_map as _impl

        return _impl(
            diploid_genotypes=self.get_all_genotypes(unordered=self.unordered),
            haploid_genotypes=self.get_all_haploid_genotypes(),
            n_glabs=len(self.gamete_labels or ["default"]),
            n_slabs=n_slabs,
            gamete_modifiers=gamete_modifiers,
        )

    def build_zygote_map(
        self,
        zygote_modifiers: Optional[list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
        n_slabs: int = 1,
    ) -> NDArray[np.float64]:
        """Build the gamete pair → diploid genotype map for this species.

        When *zygote_modifiers* is None, returns the Mendelian baseline.

        Args:
            zygote_modifiers: Optional modifier callables to apply.
            n_slabs: Number of somatic slabs.  When > 1 the genotype axis is
                tiled so that each base genotype appears once per slab.
        """
        self = cast(Species, self)
        from natal.frontend.genetics import initialize_zygote_map as _impl

        return _impl(
            haploid_genotypes=self.get_all_haploid_genotypes(),
            diploid_genotypes=self.get_all_genotypes(unordered=self.unordered),
            n_glabs=len(self.gamete_labels or ["default"]),
            n_slabs=n_slabs,
            unordered=self.unordered,
            zygote_modifiers=zygote_modifiers,
        )

    def get_config_blueprint(self) -> SpeciesConfigBlueprint:
        """Return species-derived arrays cached for population construction.

        Built once per species and cached — genotype / gamete maps, the
        offspring probability tensor, and genotype compatibility arrays.

        This is the single baseline acquisition entry.  A cached baseline
        is reused only while the dependency content snapshot is unchanged
        (structure edits invalidate it lazily) and its array contents are
        unchanged; otherwise the baseline is rebuilt.  The rebuild
        replaces the cache entry wholesale — a failed rebuild leaves no
        stale baseline reachable.

        PopulationBuilder and PopulationBuilder call this during build to avoid
        recomputing species-level arrays on every construction.

        Returns:
            Dict with keys ``n_ztypes`` (int), ``n_gtypes``
            (int), ``n_glabs`` (int), ``zygotes_to_gametes_map``
            (ndarray), ``gametes_to_zygotes_map`` (ndarray),
            ``offspring_tensor`` (ndarray), and compatibility arrays
            (ndarray).
        """
        self = cast(Species, self)
        # Gate the baseline behind structure completeness even on cache
        # hits: an incomplete edit must fail the next computation instead
        # of silently reusing results computed from a complete structure.
        self.validate_structure()
        snapshot = self._structure_dependency_snapshot()
        cached = self.config_blueprint
        if (
            cached is not None
            and self.blueprint_snapshot == snapshot
            and self.blueprint_content_snapshot == self._blueprint_content_snapshot(cached)
        ):
            return cached

        # Rebuild replaces the cache entry; a failure here must not leave
        # the previous baseline reachable through this entry.
        self.config_blueprint = None
        self.blueprint_snapshot = None
        self.blueprint_content_snapshot = None

        from natal.frontend.genetics.matrices import recompute_offspring_tensor

        genotypes = self.get_all_genotypes(unordered=self.unordered)
        haplotypes = self.get_all_haploid_genotypes()
        n_glabs = len(self.gamete_labels or ["default"])
        n_slabs = len(self.somatic_labels or ["default"])
        n_g = len(genotypes)
        n_hg = len(haplotypes)

        z2g = self.build_gamete_map(n_slabs=n_slabs)
        g2z = self.build_zygote_map(n_slabs=n_slabs)

        meiosis_f = cast(NDArray[np.float64], z2g[0])
        meiosis_m = cast(NDArray[np.float64], z2g[1])

        n_ztypes = n_g * n_slabs
        n_gtypes = n_hg * n_glabs

        offspring = recompute_offspring_tensor(z2g, g2z)

        f_compat = meiosis_f.sum(axis=1)
        m_compat = meiosis_m.sum(axis=1)

        # Sex constraints come from the genetic structure (valid
        # XX/XY, ZW/ZZ pairings), never from the compatibility row sums.
        import numpy as _np

        female_only = _np.zeros(n_g, dtype=_np.bool_)
        male_only = _np.zeros(n_g, dtype=_np.bool_)
        if self.get_sex_chromosome_groups():
            for gi, genotype in enumerate(genotypes):
                constraint = self.classify_genotype_sex(genotype)
                if constraint == "female":
                    female_only[gi] = True
                elif constraint == "male":
                    male_only[gi] = True

        blueprint: SpeciesConfigBlueprint = {
            "n_genotypes": n_g,
            "n_ztypes": n_ztypes,
            "n_gtypes": n_gtypes,
            "n_glabs": n_glabs,
            "n_slabs": n_slabs,
            "zygotes_to_gametes_map": z2g,
            "gametes_to_zygotes_map": g2z,
            "offspring_tensor": offspring,
            "female_ztype_compatibility": f_compat,
            "male_ztype_compatibility": m_compat,
            "female_only_by_sex_chrom": female_only,
            "male_only_by_sex_chrom": male_only,
        }
        self._validate_blueprint_content(blueprint)
        self.config_blueprint = blueprint
        self.blueprint_snapshot = snapshot
        self.blueprint_content_snapshot = self._blueprint_content_snapshot(blueprint)
        return self.config_blueprint

    def _structure_dependency_snapshot(self) -> object:
        """Content snapshot of every Species input the baseline depends on.

        Returns an opaque token used only for exact ``==`` comparison
        against the stored snapshot.  It holds independent values —
        chromosome/locus/allele catalogs with order, locus positions,
        recombination rates, sex-chromosome types, label names, and the
        unordered flag — never shared array views, so any setter, bulk
        entry, or write through a shared view produces a different token
        on the next acquisition.
        """
        self = cast(Species, self)
        chromosomes: list[object] = []
        for chrom in self.chromosomes:
            recomb = chrom.recombination_map if len(chrom.loci) >= 2 else None
            rates = None
            if recomb is not None:
                raw = np.asarray(recomb)
                if tuple(recomb.loci_names) != tuple(locus.name for locus in chrom.loci):
                    raise ValueError(f"Chromosome {chrom.name!r}: recombination locus mapping differs from the structure")
                if raw.shape != (len(chrom.loci) - 1,):
                    raise ValueError(f"Chromosome {chrom.name!r}: invalid recombination map shape")
                if not np.all(np.isfinite(raw)) or np.any((raw < 0) | (raw > 0.5)):
                    raise ValueError(f"Chromosome {chrom.name!r}: recombination rates must be finite and in [0, 0.5]")
                rates = (tuple(recomb.loci_names), raw.tobytes())
            chromosomes.append((
                id(chrom), chrom.name,
                str(chrom.sex_type),
                tuple(
                    (id(locus), locus.name, locus.position, tuple((id(g), g.name) for g in locus.alleles))
                    for locus in chrom.loci
                ),
                rates,
            ))
        return (
            self.unordered,
            tuple(self.gamete_labels or ()),
            tuple(self.somatic_labels or ()),
            tuple(chromosomes),
        )

    @staticmethod
    def _blueprint_content_snapshot(blueprint: SpeciesConfigBlueprint) -> object:
        """Detect all caller edits, including legal probabilities and replaced fields.

        Store exact owned bytes rather than views: range checks alone cannot
        distinguish an untouched baseline from a valid but altered matrix.
        """
        values: list[object] = []
        for key, value in sorted(blueprint.items()):
            if isinstance(value, np.ndarray):
                array = cast(NDArray[np.float64], value)
                values.append((key, array.shape, array.dtype.str, array.tobytes()))
            else:
                values.append((key, value))
        return tuple(values)

    def _validate_blueprint_content(self, blueprint: SpeciesConfigBlueprint) -> None:
        """Reject corrupted baseline values before use.

        Guards both the fresh and the cache-hit path against values made
        illegal through a shared array view: finiteness, axis shapes
        aligned to the catalogs, and probability ranges are verified
        instead of trusting the snapshot comparison alone.
        """
        import numpy as np

        n_z = int(blueprint["n_ztypes"])
        n_g = int(blueprint["n_gtypes"])
        expected: dict[str, tuple[int, ...]] = {
            "zygotes_to_gametes_map": (2, n_z, n_g),
            "gametes_to_zygotes_map": (n_g, n_g, n_z),
            "offspring_tensor": (n_z, n_z, n_z),
            "female_ztype_compatibility": (n_z,),
            "male_ztype_compatibility": (n_z,),
        }
        for key, shape in expected.items():
            # The TypedDict unions scalar catalog fields in; every key in
            # `expected` maps to an ndarray field.
            arr = np.asarray(cast(NDArray[np.float64], blueprint[key]))
            if arr.shape != shape:
                raise RuntimeError(
                    f"Species baseline field '{key}' has shape {arr.shape}, "
                    f"expected {shape}"
                )
            if arr.size and not bool(np.all(np.isfinite(arr))):
                raise RuntimeError(
                    f"Species baseline field '{key}' contains non-finite values"
                )
            if arr.size and float(arr.min()) < 0.0:
                raise RuntimeError(
                    f"Species baseline field '{key}' contains negative values"
                )
        for key in ("zygotes_to_gametes_map", "gametes_to_zygotes_map"):
            arr = np.asarray(blueprint[key])
            if arr.size and float(arr.max()) > 1.0 + 1e-9:
                raise RuntimeError(
                    f"Species baseline field '{key}' contains probabilities > 1"
                )
        n_genotypes = int(blueprint["n_genotypes"])
        n_slabs = int(blueprint["n_slabs"])
        if n_z != n_genotypes * n_slabs:
            raise RuntimeError(
                f"Species baseline catalogs disagree: n_ztypes={n_z} but "
                f"n_genotypes={n_genotypes} x n_slabs={n_slabs}"
            )
