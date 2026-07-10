"""Cytoplasmic inheritance and slab-based presets.

Public module — provides CytoplasmicPreset, Wolbachia, and TransgenicBackground.
"""

# pyright: reportPrivateUsage=false

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from natal.genetics import Genotype, Species
from natal.modifiers.module import GameteModifier, ZygoteModifier

from ._base import GeneticPreset
from ._types import PresetFitnessPatch

if TYPE_CHECKING:
    from natal.population.base import BasePopulation


# ---------------------------------------------------------------------------
# Slab-aware presets
# ---------------------------------------------------------------------------


class CytoplasmicPreset(GeneticPreset):
    """Base class for maternally-inherited cytoplasmic elements.

    Child slab = mother slab regardless of father.  The mechanism:
    1. *Gamete tagging* happens externally during slab expansion
       (via ``build_population_config``) — the
       non-default glab/slab pairs are auto-detected by convention.
       ``gamete_modifier`` returns ``None`` (no per-modifier tagging).
    2. ``apply_zygote_redirect`` (called during zygote map expansion)
       redirects tagged gamete pairs from slab-0 to the correct child slab.

    Subclasses must provide ``_maternal_map`` — a dict mapping
    ``{maternal_slab_name: glab_name}``.  Each maternal slab that
    should be heritable gets a unique glab for tagging.

    Example (Wolbachia):
        _maternal_map = {"infected": "wolbachia"}
    """

    _maternal_map: dict[str, str] = {}  # {slab_name: glab_name}

    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Tag maternal gametes: default-glab → *glab_name* for matching slabs.

        Only female gametes from mothers in the mapped slab are tagged.
        """
        if not self._maternal_map:
            return None

        def modifier_func(*_args: object, **_kwargs: object) -> Dict[
            Tuple[int, int], Dict[int, float]
        ]:
            registry = population.registry
            z2g = population.config.zygotes_to_gametes_map
            result: Dict[Tuple[int, int], Dict[int, float]] = {}

            for slab_name, glab_name in self._maternal_map.items():
                for ztype_idx, (_genotype, slab) in enumerate(
                    cast(list[tuple[object, str]], registry.index_to_ztype)
                ):
                    if slab != slab_name:
                        continue
                    row = z2g[0, ztype_idx, :]
                    dist: Dict[int, float] = {}
                    for gtype_idx in range(len(row)):
                        val = float(row[gtype_idx])
                        if val > 0:
                            hg, glab = registry.index_to_gtype[gtype_idx]
                            if glab == "default":
                                # Use glab index arithmetic (matches original tag_maternal_gametes)
                                glab_idx = registry.glab_to_index.get(glab_name)
                                if glab_idx is None:
                                    continue
                                hg_idx = registry.haplo_to_index.get(hg)
                                if hg_idx is None:
                                    continue
                                dst = hg_idx * len(registry.glab_labels) + glab_idx
                                dist[dst] = dist.get(dst, 0.0) + val
                            else:
                                dist[gtype_idx] = dist.get(gtype_idx, 0.0) + val
                    if dist:
                        result[(0, ztype_idx)] = dist

            return result

        return modifier_func  # type: ignore[return-type]

    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Redirect zygotes: tagged maternal gamete + any paternal → target slab."""
        if not self._maternal_map:
            return None

        def modifier_func(*_args: object, **_kwargs: object) -> Dict[
            Tuple[int, int], Dict[int, float]
        ]:
            registry = population.registry
            g2z = population.config.gametes_to_zygotes_map
            result: Dict[Tuple[int, int], Dict[int, float]] = {}
            default_slab = registry.slab_labels[0]
            n_gtypes = g2z.shape[0]

            for slab_name, glab_name in self._maternal_map.items():
                for c1 in range(n_gtypes):
                    _, c1_glab = registry.index_to_gtype[c1]
                    if c1_glab != glab_name:
                        continue
                    for c2 in range(n_gtypes):
                        row = g2z[c1, c2]
                        if row.sum() == 0:
                            continue
                        dist: Dict[int, float] = {}
                        for ztype_idx in range(len(row)):
                            val = float(row[ztype_idx])
                            if val <= 0:
                                continue
                            gt, slab = cast(
                                tuple[object, str], registry.index_to_ztype[ztype_idx]
                            )
                            if slab == default_slab:
                                dst_z = registry.ztype_index(
                                    cast(Genotype, gt), slab_name
                                )
                                dist[dst_z] = dist.get(dst_z, 0.0) + val
                            else:
                                dist[ztype_idx] = dist.get(ztype_idx, 0.0) + val
                        if dist:
                            result[(c1, c2)] = dist

            return result

        return modifier_func  # type: ignore[return-type]

    @staticmethod
    def apply_zygote_redirect(
        z2g_expanded: NDArray[np.float64],
        glab_name: str,
        slab_name: str,
        gamete_labels: List[str],
        somatic_labels: List[str],
        n_slabs: int,
        n_genotypes_raw: int,
        n_hg: int,
        n_glabs: int,
    ) -> None:
        """Redirect zygote columns: glab-tagged maternal gametes → target slab.

        Looks up *glab_name* and *slab_name* in the label lists (not
        the registry — this runs in build_population_config which has
        no registry access).  No-op if either label is missing.
        """
        if glab_name not in gamete_labels or slab_name not in somatic_labels:
            return
        glab_idx = gamete_labels.index(glab_name)
        slab_idx = somatic_labels.index(slab_name)
        for g_raw in range(n_genotypes_raw):
            z_dst = g_raw * n_slabs + slab_idx
            z_src = g_raw * n_slabs + 0
            for hg_f in range(n_hg):
                hl_f = hg_f * n_glabs + glab_idx
                for hg_m in range(n_hg):
                    for gm in range(n_glabs):
                        hl_m = hg_m * n_glabs + gm
                        val = z2g_expanded[hl_f, hl_m, z_src]
                        if val > 0:
                            z2g_expanded[hl_f, hl_m, z_dst] += val
                            z2g_expanded[hl_f, hl_m, z_src] = 0.0

    @staticmethod
    def tag_maternal_gametes(
        z2g_expanded: NDArray[np.float64],
        gamete_labels: List[str],
        somatic_labels: List[str],
        n_genotypes: int,
        n_gtypes: int,
        n_glabs: int,
        n_slabs: int,
    ) -> None:
        """Tag maternal gametes so non-default glabs are inherited maternally.

        Operates on the **slab-expanded** z2g map (post-expansion, shape
        ``(2, G×S, HL)``).  For each matching glab/slab index pair
        (glab[i] ↔ slab[i] for i >= 1), copies the default-glab female
        gamete probabilities to the non-default glab column — and zeros
        out the original — so that only mothers carrying the matching
        slab produce gametes with the non-default glab.

        This method should be called **after** slab expansion and **before**
        redirect_zygotes.

        Args:
            z2g_expanded: Slab-expanded genotype-to-gamete map, shape
                ``(2, G×S, HL)``.  Modified in-place.
            gamete_labels: Ordered gamete label names.
            somatic_labels: Ordered somatic label names.
            n_genotypes: Pre-expansion genotype count ``G``.
            n_gtypes: Number of haploid genotype types (HL / n_glabs).
            n_glabs: Number of gamete label types.
            n_slabs: Number of somatic label types.
        """
        if not gamete_labels or not somatic_labels or n_glabs < 2 or n_slabs < 2:
            return
        for idx in range(1, min(n_slabs, n_glabs)):
            if idx >= len(gamete_labels) or idx >= len(somatic_labels):
                continue
            for g_raw in range(n_genotypes):
                z_target = g_raw * n_slabs + idx
                for hg_idx in range(n_gtypes):
                    src = hg_idx * n_glabs + 0   # default glab
                    dst = hg_idx * n_glabs + idx
                    z2g_expanded[0, z_target, dst] = z2g_expanded[0, z_target, src]
                    z2g_expanded[0, z_target, src] = 0.0

    @staticmethod
    def redirect_zygotes(
        g2z_expanded: NDArray[np.float64],
        gamete_labels: List[str],
        somatic_labels: List[str],
        n_genotypes: int,
        n_gtypes: int,
        n_glabs: int,
        n_slabs: int,
    ) -> None:
        """Redirect zygote columns: glab-tagged gamete pairs → matching child slab.

        Iterates matching glab/slab index pairs (glab[i] ↔ slab[i] for
        i >= 1) and calls :meth:`apply_zygote_redirect` for each pair.
        That method moves zygote probabilities from the default slab-0
        column to the matching child slab column when the maternal gamete
        carries the glab tag — enforcing strict maternal inheritance.

        This method should be called **after** slab expansion and
        ``tag_maternal_gametes``.

        Args:
            g2z_expanded: Slab-expanded gametes-to-zygote map, shape
                ``(HL, HL, G×S)``.  Modified in-place.
            gamete_labels: Ordered gamete label names.
            somatic_labels: Ordered somatic label names.
            n_genotypes: Pre-expansion genotype count ``G``.
            n_gtypes: Number of haploid genotype types (HL / n_glabs).
            n_glabs: Number of gamete label types.
            n_slabs: Number of somatic label types.
        """
        if not gamete_labels or not somatic_labels or n_glabs < 2 or n_slabs < 2:
            return
        for idx in range(1, min(n_slabs, n_glabs)):
            if idx >= len(gamete_labels) or idx >= len(somatic_labels):
                continue
            CytoplasmicPreset.apply_zygote_redirect(
                g2z_expanded, gamete_labels[idx], somatic_labels[idx],
                gamete_labels, somatic_labels,
                n_slabs, n_genotypes,
                n_gtypes, n_glabs,
            )


class Wolbachia(CytoplasmicPreset):
    """Maternally-inherited endosymbiont.  Infected mothers pass the
    infection to all offspring regardless of the father.

    Requires Species with:
      - gamete_labels including ``"wolbachia"``
      - somatic_labels including ``"normal"``, ``"infected"``
    """

    def __init__(
        self,
        name: str,
        infected_slab: str = "infected",
        normal_slab: str = "normal",
        viability_scaling: float = 1.0,
        fecundity_scaling: Optional[float] = None,
        species: Optional[Species] = None,
        priority: int = 0,
    ):
        """Initialize a Wolbachia cytoplasmic preset.

        Args:
            name: Preset name.
            infected_slab: Somatic slab label for infected individuals.
            normal_slab: Somatic slab label for uninfected individuals.
            viability_scaling: Viability multiplier for infected carriers.
            fecundity_scaling: Fecundity multiplier for infected carriers.
                ``None`` means no fecundity effect.
            species: Optional species for validation.
            priority: Modifier and fitness application priority.
        """
        super().__init__(name=name, species=species, priority=priority)
        self._maternal_map = {infected_slab: "wolbachia"}
        self.infected_slab = infected_slab
        self.normal_slab = normal_slab
        self.viability_scaling = viability_scaling
        self.fecundity_scaling = fecundity_scaling

    def fitness_patch(self) -> PresetFitnessPatch:
        """Build fitness patch applying viability and fecundity scaling.

        Returns:
            A fitness patch dict with optional ``viability_per_slab`` and
            ``fecundity_per_slab`` entries for the infected slab.
        """
        patch: PresetFitnessPatch = {}
        patch['viability_per_slab'] = {self.infected_slab: self.viability_scaling}
        if self.fecundity_scaling is not None:
            patch['fecundity_per_slab'] = {self.infected_slab: self.fecundity_scaling}
        return patch


class TransgenicBackground(GeneticPreset):
    """Fitness scaling for a transgenic background slab.

    Applies fecundity and/or viability scaling to individuals carrying
    the *tg_slab* somatic label.  Does NOT implement outcrossing
    clearance — that requires a separate inheritance mechanism.
    """

    def __init__(
        self,
        name: str,
        tg_slab: str,
        wt_slab: str = "WT_bg",
        fecundity_scaling: float = 1.0,
        viability_scaling: Optional[float] = None,
        species: Optional[Species] = None,
        priority: int = 0,
    ):
        """Initialize a TransgenicBackground preset.

        Args:
            name: Preset name.
            tg_slab: Somatic slab label for transgenic individuals.
            wt_slab: Somatic slab label for wild-type background.
            fecundity_scaling: Fecundity multiplier for transgenic carriers.
            viability_scaling: Optional viability multiplier for transgenic
                carriers. ``None`` means no viability effect.
            species: Optional species for validation.
            priority: Modifier and fitness application priority.
        """
        super().__init__(name=name, species=species, priority=priority)
        self.tg_slab = tg_slab
        self.wt_slab = wt_slab
        self.fecundity_scaling = fecundity_scaling
        self.viability_scaling = viability_scaling

    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Return no gamete modifier — transgenic background is slab-only."""
        return None

    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Return no zygote modifier — transgenic background is slab-only."""
        return None

    def fitness_patch(self) -> PresetFitnessPatch:
        """Build fitness patch applying fecundity and optional viability scaling.

        Returns:
            A fitness patch dict with ``fecundity_per_slab`` and optionally
            ``viability_per_slab`` entries for the transgenic slab.
        """
        patch: PresetFitnessPatch = {}
        patch['fecundity_per_slab'] = {self.tg_slab: self.fecundity_scaling}
        if self.viability_scaling is not None:
            patch['viability_per_slab'] = {self.tg_slab: self.viability_scaling}
        return patch
