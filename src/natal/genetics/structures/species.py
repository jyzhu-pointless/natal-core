"""Species — top-level genetic architecture structure."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from ._base import GeneticStructure
from ._construction import SpeciesConstructionMixin
from ._enumeration import SpeciesEnumerationMixin
from ._mapping import SpeciesMappingMixin
from ._pattern import SpeciesPatternMixin
from ._types import SexChromosomeType
from .chromosome import Chromosome
from .locus import Locus

if TYPE_CHECKING:
    from ..entities.gene import Gene
    from ..entities.haplotype import (
        HaploidGenome,  # noqa: F401 — used in GeneticStructure['HaploidGenome']
    )
    from ._registry import ChildStructureRegistry


class SpeciesConfigBlueprint(TypedDict):
    """Cached species-level arrays shared across population constructions.

    Built once per species by :meth:`Species.get_config_blueprint` and
    consumed by ``Configurator`` / ``PopulationBuilder``.
    """

    n_genotypes: int
    n_ztypes: int
    n_gtypes: int
    n_glabs: int
    n_slabs: int
    zygotes_to_gametes_map: NDArray[np.float64]
    gametes_to_zygotes_map: NDArray[np.float64]
    offspring_tensor: NDArray[np.float64]
    female_ztype_compatibility: NDArray[np.float64]
    male_ztype_compatibility: NDArray[np.float64]


class Species(
    SpeciesConstructionMixin,
    SpeciesEnumerationMixin,
    SpeciesMappingMixin,
    SpeciesPatternMixin,
    GeneticStructure['HaploidGenome'],
):
    """
    Represents the complete genetic architecture defined by chromosomes.

    A Species is the top-level structure that contains multiple Chromosomes,
    each representing a chromosome with its loci and recombination rates.

    Attributes:
        child_structures (ChildStructureRegistry[Chromosome]): Registry of chromosomes.
        structure_cache (Dict[type, Dict[str, GeneticStructure[Any]]]): Species-scoped
            structure cache grouped by structure type.
        gamete_labels (List[str]): Optional labels used to identify gamete categories.

    This class is also exported as GenomeTemplate for backward compatibility.
    """
    child_structure_type = Chromosome
    child_structures: ChildStructureRegistry[Chromosome]
    sex_chromosome_groups: Optional[Dict[str, List[Chromosome]]]
    valid_sex_genotypes: Optional[List[Tuple[Chromosome, Chromosome]]]

    def __init__(
        self,
        name: str,
        chromosomes: Optional[List[Chromosome]] = None,
        gamete_labels: Optional[list[str]] = None,
        somatic_labels: Optional[list[str]] = None,
        unordered: bool = True,
    ):
        self._structure_cache: Dict[type, Dict[str, GeneticStructure[Any]]] = {}
        self._gene_index_cache: Optional[Dict[str, Gene]] = None
        self._unordered = unordered

        super().__init__(name, parent=None, species=None)

        self._species = self

        if chromosomes:
            for chrom in chromosomes:
                self.add_chromosome(chrom)

        if gamete_labels is not None:
            from natal.utils.helpers import validate_name
            for glab in gamete_labels:
                if not validate_name(glab):
                    raise ValueError(
                        f"Invalid gamete label {glab!r}. "
                        f"Labels must match [A-Za-z0-9_]+."
                    )
            self._gamete_labels = list(gamete_labels)
        else:
            self._gamete_labels: List[str] = []

        if somatic_labels is not None:
            from natal.utils.helpers import validate_name
            for slab in somatic_labels:
                if not validate_name(slab):
                    raise ValueError(
                        f"Invalid somatic label {slab!r}. "
                        f"Labels must match [A-Za-z0-9_]+."
                    )
            self._somatic_labels = list(somatic_labels)
        else:
            self._somatic_labels: List[str] = []

        self.config_blueprint: Optional[SpeciesConfigBlueprint] = None

    @property
    def gamete_labels(self) -> List[str]:
        """Return the list of gamete labels for this species."""
        return self._gamete_labels

    @gamete_labels.setter
    def gamete_labels(self, labels: List[str]) -> None:
        self._gamete_labels = list(labels)

    @property
    def somatic_labels(self) -> List[str]:
        """Return the list of somatic labels for this species."""
        return self._somatic_labels

    @somatic_labels.setter
    def somatic_labels(self, labels: List[str]) -> None:
        self._somatic_labels = list(labels)

    @property
    def entity_type(self):
        """Lazy import to avoid circular dependency."""
        from ..entities.haplotype import HaploidGenome
        return HaploidGenome

    def clear_structure_cache(self) -> None:
        """Clear all structure caches for this Species."""
        self._structure_cache.clear()
        self._invalidate_gene_index_cache()

    @property
    def structure_cache(self) -> Dict[type, Dict[str, GeneticStructure[Any]]]:
        """Public accessor for species-scoped structure caches."""
        return self._structure_cache

    def clear_entity_cache(self) -> None:
        """Clear all entity caches for this Species."""
        from ..entities._base import GeneticEntity
        GeneticEntity.clear_species_cache(id(self))

    def clear_all_caches(self) -> None:
        """Clear both structure and entity caches for this Species."""
        self.clear_structure_cache()
        self.clear_entity_cache()

    def _invalidate_gene_index_cache(self) -> None:
        """Invalidate species-level gene name lookup cache."""
        self._gene_index_cache = None

    def invalidate_gene_index_cache(self) -> None:
        """Public wrapper for invalidating the species gene-index cache."""
        self._invalidate_gene_index_cache()

    @property
    def chromosomes(self) -> List[Chromosome]:
        """Returns the list of chromosomes in this species."""
        return self.child_structures.all

    @property
    def linkages(self) -> List[Chromosome]:
        """Alias for chromosomes (backward compatibility)."""
        return self.chromosomes

    @property
    def sex_chromosomes(self) -> List[Chromosome]:
        """Returns all sex chromosomes"""
        return [c for c in self.chromosomes if c.is_sex_chromosome]

    @property
    def unordered(self) -> bool:
        """True means maternal/paternal chromosome ordering is irrelevant.

        When True (default), genotype enumeration uses canonical unordered
        genotype space.  Set ``unordered=False`` to track chromosome
        parentage explicitly.
        """
        return self._unordered

    @property
    def autosomes(self) -> List[Chromosome]:
        """Returns all autosomes"""
        return [c for c in self.chromosomes if c.is_autosome]

    @property
    def sex_system(self) -> Optional[str]:
        """
        Returns the sex determination system ('XY', 'ZW', or None).

        Automatically inferred from Chromosome.sex_type. Raises an error if multiple systems are found.
        """
        systems: Set[str] = set()
        for chrom in self.chromosomes:
            if chrom.sex_system:
                systems.add(chrom.sex_system)

        if len(systems) == 0:
            return None
        elif len(systems) == 1:
            return systems.pop()
        else:
            raise ValueError(
                f"Multiple sex chromosome systems detected: {systems}. "
                f"A species should only have one sex determination system."
            )

    @property
    def gene_index(self) -> Dict[str, Gene]:
        """Returns a mapping from gene names to gene instances."""
        return self.build_gene_index()

    def build_sex_chromosome_groups(self) -> Dict[str, List[Chromosome]]:
        """
        Automatically build sex_chromosome_groups from Chromosome.sex_type.

        Returns:
            Sex chromosome group dictionary, keys are system names like 'XY' or 'ZW'
        """
        groups: Dict[str, List[Chromosome]] = {}
        for chrom in self.chromosomes:
            system = chrom.sex_system
            if system:
                if system not in groups:
                    groups[system] = []
                groups[system].append(chrom)
        return groups

    def build_valid_sex_genotypes(self) -> List[Tuple[Chromosome, Chromosome]]:
        """
        Automatically infer valid sex chromosome genotype combinations from Chromosome.sex_type.

        Rules include:
            - XY system: X can come from either parent, Y is paternal only
                    -> Valid combinations: (X, X), (X, Y)
            - ZW system: Z can come from either parent, W is maternal only
                    -> Valid combinations: (Z, Z), (W, Z)

        Returns:
            List of valid (maternal_chrom, paternal_chrom) combinations
        """
        valid_combos: List[Tuple[Chromosome, Chromosome]] = []

        system_chroms: Dict[str, Dict[str, Chromosome]] = {}
        for chrom in self.chromosomes:
            if not chrom.is_sex_chromosome:
                continue
            system = chrom.sex_system
            if system is None:
                continue
            if system not in system_chroms:
                system_chroms[system] = {}
            system_chroms[system][chrom.sex_type.value] = chrom

        for system, chroms in system_chroms.items():
            if system == 'XY':
                x_chrom = chroms.get('X')
                y_chrom = chroms.get('Y')
                if x_chrom:
                    valid_combos.append((x_chrom, x_chrom))
                    if y_chrom:
                        valid_combos.append((x_chrom, y_chrom))
            elif system == 'ZW':
                z_chrom = chroms.get('Z')
                w_chrom = chroms.get('W')
                if z_chrom:
                    valid_combos.append((z_chrom, z_chrom))
                    if w_chrom:
                        valid_combos.append((w_chrom, z_chrom))

        return valid_combos

    def add_chromosome(
        self,
        chrom_or_name: Union[Chromosome, str],
        loci: Optional[List[Locus]] = None,
        sex_type: Optional[Union[SexChromosomeType, str]] = None
    ) -> Chromosome:
        """
        Add a chromosome to this species.

        Args:
            chrom_or_name: Either a Chromosome instance or a name to create a new one.
            loci: Optional list of loci (only used when creating new Chromosome by name).
            sex_type: Optional sex chromosome type (X', 'Y', 'Z', 'W', None).

        Returns:
            The added Chromosome instance.
        """
        assert isinstance(chrom_or_name, (Chromosome, str)), \
            f"Expected Chromosome instance or str, got {type(chrom_or_name).__name__}"

        if isinstance(chrom_or_name, str):
            created = self.add(chrom_or_name, loci=loci, sex_type=sex_type)
            assert isinstance(created, Chromosome), \
                f"Expected add() to return Chromosome, got {type(created).__name__}"
            chrom = created
        else:
            chrom = chrom_or_name
            if sex_type is not None:
                chrom.sex_type = sex_type
            if chrom.name not in self.child_structures:
                self.child_structures.register(chrom)

        self.invalidate_gene_index_cache()
        return chrom

    def add_linkage(
        self,
        linkage_or_name: Union[Chromosome, str],
        loci: Optional[List[Locus]] = None
    ) -> Chromosome:
        """Alias for add_chromosome (backward compatibility)."""
        return self.add_chromosome(linkage_or_name, loci=loci)

    def remove_chromosome(self, chrom_or_name: Union[Chromosome, str]) -> None:
        """
        Remove a chromosome from this species.

        Args:
            chrom_or_name: Either a Chromosome instance or a name.
        """
        if isinstance(chrom_or_name, str):
            name = chrom_or_name
        else:
            name = chrom_or_name.name

        if name in self.child_structures:
            self.child_structures.unregister(name)
            self._invalidate_gene_index_cache()

    def remove_linkage(self, linkage_or_name: Union[Chromosome, str]) -> None:
        """Alias for remove_chromosome (backward compatibility)."""
        return self.remove_chromosome(linkage_or_name)

    def get_all_loci(self) -> List[Locus]:
        """Returns all loci across all chromosomes."""
        all_loci: List[Locus] = []
        for chrom in self.chromosomes:
            all_loci.extend(chrom.loci)
        return all_loci

    def get_locus(self, name: str) -> Optional[Locus]:
        """
        Get a locus by name across all chromosomes.

        Args:
            name: Name of the locus.

        Returns:
            The Locus instance or None if not found.
        """
        for chrom in self.chromosomes:
            for locus in chrom.loci:
                if locus.name == name:
                    return locus
        return None

    def get_chromosome(self, name: str) -> Optional[Chromosome]:
        """
        Get a chromosome by name.

        Args:
            name: Name of the chromosome.

        Returns:
            The Chromosome instance or None if not found.
        """
        if name in self.child_structures:
            return self.child_structures.get(name)
        return None

    def get_gene(self, name: str) -> Optional[Gene]:
        """
        Get a gene by name across all loci.

        Args:
            name: Name of the gene.

        Returns:
            The Gene instance or None if not found.

        Raises:
            ValueError: If duplicate gene names exist in the species.
        """
        try:
            gene_index = self.build_gene_index()
            return gene_index.get(name)
        except ValueError:
            raise

    def has_gene(self, name: str) -> bool:
        """
        Check if a gene with the given name exists in the species.

        Args:
            name: Name of the gene to check.

        Returns:
            True if the gene exists, False otherwise.

        Raises:
            ValueError: If duplicate gene names exist in the species.
        """
        try:
            gene_index = self.build_gene_index()
            return name in gene_index
        except ValueError:
            raise

    def get_linkage(self, name: str) -> Optional[Chromosome]:
        """Alias for get_chromosome (backward compatibility)."""
        return self.get_chromosome(name)

    def build_gene_index(self) -> Dict[str, Gene]:
        """
        Build a lookup index from gene name to Gene object.

        Returns:
            Dict mapping gene name to Gene instance.

        Raises:
            ValueError: If duplicate gene names exist in the species.
        """
        if self._gene_index_cache is not None:
            return self._gene_index_cache

        gene_index: Dict[str, Gene] = {}
        for chrom in self.chromosomes:
            for locus in chrom.loci:
                for gene in locus.alleles:
                    if gene.name in gene_index:
                        raise ValueError(
                            f"Duplicate gene name '{gene.name}' found in species. "
                            f"Gene names must be unique for string-based lookups. "
                            f"Found at locus '{gene.locus.name}' and '{gene_index[gene.name].locus.name}'."
                        )
                    gene_index[gene.name] = gene
        self._gene_index_cache = gene_index
        return gene_index

    def __repr__(self):
        chrom_strs: List[str] = []
        for chrom in self.chromosomes:
            loci_names = [locus.name for locus in chrom.loci]
            chrom_strs.append(f"'{chrom.name}': {loci_names}")
        return f"Species({self.name!r}, {{{', '.join(chrom_strs)}}})"

    def __iter__(self):
        return iter(self.chromosomes)

    def __len__(self):
        return len(self.chromosomes)
