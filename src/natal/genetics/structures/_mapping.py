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
    from ..entities.genotype import Genotype
    from ..entities.haplotype import HaploidGenotype
    from .species import Species, SpeciesConfigBlueprint
else:
    Species = object  # runtime stand-in for cast()


class SpeciesMappingMixin:
    """Mapping methods for Species — genotype ordering, gamete/zygote maps, config blueprint.

    Provides unordered genotype canonicalisation and methods to build the
    genotype-to-gamete and gamete-to-zygote transition maps used by the
    simulation engine.
    """

    def unordered_genotype(
        self,
        hg1: HaploidGenotype,
        hg2: HaploidGenotype,
    ) -> Genotype:
        """Return a canonical Genotype where maternal/paternal order is irrelevant.

        Canonicalises per-locus: at each locus the maternal allele has the
        smaller :meth:`Locus.allele_index`.  When individual alleles must be
        swapped between the two haploid genomes (multi-locus free combination)
        new :class:`HaploidGenotype` objects are assembled so that every
        genotype with the same per-locus allele composition collapses to the
        same canonical form.
        """
        self = cast(Species, self)
        from ..entities.genotype import Genotype
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
        from natal.data import initialize_gamete_map as _impl

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
        from natal.data import initialize_zygote_map as _impl

        return _impl(
            haploid_genotypes=self.get_all_haploid_genotypes(),
            diploid_genotypes=self.get_all_genotypes(unordered=self.unordered),
            n_glabs=len(self.gamete_labels or ["default"]),
            n_slabs=n_slabs,
            unordered=True,
            zygote_modifiers=zygote_modifiers,
        )

    def get_config_blueprint(self) -> SpeciesConfigBlueprint:
        """Return species-derived arrays cached for population construction.

        Built once per species and cached — genotype / gamete maps, the
        offspring probability tensor, and genotype compatibility arrays.
        These never change at runtime.

        Configurator and PopulationBuilder call this during build to avoid
        recomputing species-level arrays on every construction.

        Returns:
            Dict with keys ``n_ztypes`` (int), ``n_gtypes``
            (int), ``n_glabs`` (int), ``zygotes_to_gametes_map``
            (ndarray), ``gametes_to_zygotes_map`` (ndarray),
            ``offspring_tensor`` (ndarray), and compatibility arrays
            (ndarray).
        """
        self = cast(Species, self)
        if self.config_blueprint is not None:
            return self.config_blueprint

        from natal.engine.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

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

        offspring = compute_offspring_probability_tensor(
            meiosis_f=meiosis_f,
            meiosis_m=meiosis_m,
            haplo_to_genotype_map=g2z,
            n_ztypes=n_ztypes,
            n_gtypes=n_gtypes,
        )

        f_compat = meiosis_f.sum(axis=1)
        m_compat = meiosis_m.sum(axis=1)

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
        }
        self.config_blueprint = blueprint
        return self.config_blueprint
