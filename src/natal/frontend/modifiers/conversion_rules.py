"""Unified conversion-rule declarations (CR-1 contract).

Four keyword-only rule classes replace the historical conversion-rule
surface.  They are inert declarations: validation of names, alleles and
targets against a species/registry happens once at compile time (when
the owning :class:`~natal.frontend.modifiers.gamete_conversion.GameteConversionRuleSet`
or :class:`~natal.frontend.modifiers.zygote_conversion.ZygoteConversionRuleSet`
builds against its host), while structural input validation (rate range,
filter keys, target shape, side value) happens here at declaration time.

Contracts shared by every rule:

- ``rate`` is required, finite and in ``[0, 1]``.
- ``filters`` is ``None`` (no restriction) or a plain ``Mapping[str, str]``
  of pattern keys; keys are AND-ed together, omitted keys mean
  unrestricted.  Unknown keys, keys unsupported by the rule's stage, and
  illegal patterns raise ``ValueError`` at compile time.
- ``name`` is display-only metadata.

Stage-specific filters:

=============== ========== ========== ========== =========
key             gamete     zygote     matches
=============== ========== ========== ========== =========
current         yes        yes        the entering branch's own state
parent          yes        no         the producer's ztype pattern
parent_sex      yes        no         ``female`` / ``male`` / ``both``
maternal        no         yes        the maternal gamete gtype pattern
paternal        no         yes        the paternal gamete gtype pattern
=============== ========== ========== ========== =========

Targets (whole-state rules) must be ``"[genotype or *]@[label or *]"`` —
both parts explicit; ``*`` keeps the input's corresponding part, and
``*@*`` is the identity conversion.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Optional, cast

from natal.frontend.patterns.elements._base import PatternParseError
from natal.frontend.patterns.parser import GenotypePatternParser

if TYPE_CHECKING:
    from natal.frontend.genetics.entities.haplotype import HaploidGenotype

__all__ = [
    "GameteAlleleConversionRule",
    "GameteGtypeConversionRule",
    "GameteStageRule",
    "ZygoteAlleleConversionRule",
    "ZygoteStageRule",
    "ZygoteZtypeConversionRule",
]

GAMETE_FILTER_KEYS = ("current", "parent", "parent_sex")
ZYGOTE_FILTER_KEYS = ("current", "maternal", "paternal")
SIDES = ("maternal", "paternal", "both")


def _validate_rate(rate: object) -> float:
    """Validate one conversion rate (user-declared, arbitrary input).

    Args:
        rate: The declared rate; must be a finite number in ``[0, 1]``.

    Returns:
        The validated rate as ``float``.

    Raises:
        ValueError: If *rate* is not a finite number in ``[0, 1]``.
    """
    value = float(rate)  # type: ignore[arg-call-overload]  # runtime boundary: floats from JSON/user objects
    # Reject non-finite and out-of-range values at declaration time, so the
    # compiler can treat `rate` as a probability without re-checking.
    if not math.isfinite(value):
        raise ValueError(f"rate must be finite, got {rate!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"rate must be in [0, 1], got {rate!r}")
    return value


def _validate_filters(
    filters: object, allowed_keys: tuple[str, ...], stage: str
) -> tuple[tuple[str, str], ...]:
    """Validate one filter mapping for a stage.

    Returns:
        A frozen tuple of ``(key, pattern)`` pairs (declaration order).

    Raises:
        ValueError: If a key is unknown to the stage or a pattern is empty.
    """
    if filters is None:
        return ()
    if not isinstance(filters, Mapping):  # runtime boundary: user declaration
        raise TypeError(
            f"{stage} rule filters must be a Mapping[str, str] or None, "
            f"got {type(filters).__name__}"
        )
    pairs: list[tuple[str, str]] = []
    # Pairs keep declaration order (cascade order matters); unknown keys and
    # empty patterns fail here, since stage support is known statically.
    mapping = cast("Mapping[object, object]", filters)
    items = list(mapping.items())  # runtime boundary
    for key, pattern in items:
        if not isinstance(key, str):  # runtime boundary
            raise ValueError(
                f"{stage} rule filter keys must be strings, got {key!r}"
            )
        if key not in allowed_keys:
            raise ValueError(
                f"{stage} rule filter key {key!r} is not supported; "
                f"allowed keys: {list(allowed_keys)}"
            )
        if not isinstance(pattern, str) or not pattern.strip():  # runtime boundary
            raise ValueError(
                f"{stage} rule filter {key!r} must be a non-empty pattern "
                f"string, got {pattern!r}"
            )
        pairs.append((key, pattern))
    return tuple(pairs)


def _parse_target_str(target: object, stage: str) -> str:
    """Require a string target and return it for the declaration.

    Raises:
        TypeError: If *target* is not a string.
    """
    if not isinstance(target, str):  # runtime boundary: user declaration
        raise TypeError(
            f"{stage} rule target must be a string, got {type(target).__name__}"
        )
    return target



def _validate_target_declaration(target: str, stage: str) -> None:
    """Validate the ``[genotype or *]@[label or *]`` spelling at declaration time.

    Both parts must be given explicitly; ``*`` keeps the input part.  The
    compile-time parse goes through the unified target entry; this is only
    the early, species-unbound check, sharing the same ``@`` analysis.

    Raises:
        ValueError: If the target is malformed.
    """
    try:
        GenotypePatternParser.split_conversion_target(target, stage=stage)
    except PatternParseError as exc:
        raise ValueError(str(exc)) from exc


class _RuleBase:
    """Shared declaration-time validation for conversion rules."""

    filters: Optional[Mapping[str, str]]
    name: Optional[str]
    rate: float

    def _init_common(
        self,
        rate: object,
        filters: Optional[Mapping[str, str]],
        name: Optional[str],
        allowed_keys: tuple[str, ...],
        stage: str,
    ) -> None:
        """Run the shared declaration checks (call from ``__init__``)."""
        # Structural validation only; allele/label resolution against a species
        # is deferred to compile time, when a host and registry exist.
        self.rate = _validate_rate(rate)
        self.filters = filters
        self.name = name
        # Validated now, stored as normalized pairs for the compiler.
        self.filter_pairs: tuple[tuple[str, str], ...] = _validate_filters(
            filters, allowed_keys, stage
        )


class GameteStageRule(_RuleBase):
    """Base class for gamete-stage rules (filters: current/parent/parent_sex)."""


class ZygoteStageRule(_RuleBase):
    """Base class for zygote-stage rules (filters: current/maternal/paternal)."""


class GameteGtypeConversionRule(GameteStageRule):
    """Convert a whole gamete (haploid genotype + gamete label) at ``rate``.

    The success branch adopts both target parts; the failure branch keeps
    the input state.  ``"A@*"`` changes only the haploid part, ``"*@tagged"``
    only the gamete label, and ``"*@*"`` is the identity conversion.
    """

    def __init__(
        self,
        *,
        to: object,
        rate: object,
        filters: Optional[Mapping[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize one whole-gtype gamete conversion.

        Args:
            to: Target ``"[haploid genotype or *]@[label or *]"``; the
                haploid part is a single-haplotype genotype string such
                as ``"A/B;X"``.  Must be a string.
            rate: Conversion probability in ``[0, 1]``.
            filters: ``current`` / ``parent`` / ``parent_sex`` patterns.
            name: Optional display name.

        Raises:
            ValueError: If *rate* or *filters* are invalid, or *to* is
                malformed.
        """
        self.to: str = _parse_target_str(to, "gamete gtype conversion")
        # Declaration-time validation of the "[genotype or *]@[label or *]"
        # spelling through the shared target analysis; the genotype/label
        # names are resolved against the species and registry only when the
        # owning rule set compiles through the unified target entry.
        _validate_target_declaration(self.to, "gamete gtype conversion")
        self._init_common(rate, filters, name, GAMETE_FILTER_KEYS, "gamete")

    def __repr__(self) -> str:
        """Return a readable rule identity."""
        return f"GameteGtypeConversionRule(to={self.to!r}, rate={self.rate})"


class ZygoteZtypeConversionRule(ZygoteStageRule):
    """Convert a whole zygote (diploid genotype + somatic label) at ``rate``.

    ``"A|B@I"`` replaces both parts jointly, ``"*@I"`` redirects only the
    somatic label, ``"A|B@*"`` only the genotype, and ``"*@*"`` is the
    identity conversion.
    """

    def __init__(
        self,
        *,
        to: object,
        rate: object,
        filters: Optional[Mapping[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize one whole-ztype zygote conversion.

        Args:
            to: Target ``"[diploid genotype or *]@[label or *]"``; the
                genotype part is a two-haplotype string such as
                ``"A/B;X|Y"``.  Must be a string.
            rate: Conversion probability in ``[0, 1]``.
            filters: ``current`` / ``maternal`` / ``paternal`` patterns.
            name: Optional display name.

        Raises:
            ValueError: If *rate* or *filters* are invalid, or *to* is
                malformed.
        """
        self.to: str = _parse_target_str(to, "zygote ztype conversion")
        _validate_target_declaration(self.to, "zygote ztype conversion")
        self._init_common(rate, filters, name, ZYGOTE_FILTER_KEYS, "zygote")

    def __repr__(self) -> str:
        """Return a readable rule identity."""
        return f"ZygoteZtypeConversionRule(to={self.to!r}, rate={self.rate})"


class GameteAlleleConversionRule(GameteStageRule):
    """Replace one source allele with a same-locus target allele in gametes.

    The replacement touches only copies carrying *from_allele*; the
    gamete label (glab) never changes.  The locus is located through the
    source allele, and the target must belong to the same locus — both
    are verified at compile time.
    """

    def __init__(
        self,
        *,
        from_allele: object,
        to_allele: object,
        rate: object,
        filters: Optional[Mapping[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize one allele-level gamete conversion.

        Args:
            from_allele: Source allele (gene) name; locates the locus.
                Must be a non-empty string.
            to_allele: Target allele; must belong to the same locus.
                Must be a non-empty string.
            rate: Conversion probability in ``[0, 1]``.
            filters: ``current`` / ``parent`` / ``parent_sex`` patterns.
            name: Optional display name.

        Raises:
            ValueError: If *rate* or *filters* are invalid.
            TypeError: If an allele name is not a string.
        """
        if not isinstance(from_allele, str) or not from_allele:  # runtime boundary
            raise TypeError(f"from_allele must be a non-empty string, got {from_allele!r}")
        if not isinstance(to_allele, str) or not to_allele:  # runtime boundary
            raise TypeError(f"to_allele must be a non-empty string, got {to_allele!r}")
        self.from_allele: str = from_allele
        self.to_allele: str = to_allele
        self._init_common(rate, filters, name, GAMETE_FILTER_KEYS, "gamete")

    def __repr__(self) -> str:
        """Return a readable rule identity."""
        return (
            f"GameteAlleleConversionRule({self.from_allele}->{self.to_allele}, "
            f"rate={self.rate})"
        )


class ZygoteAlleleConversionRule(ZygoteStageRule):
    """Replace one source allele per zygotic genetic copy at ``rate``.

    ``side`` selects which zygotic copies convert: ``"maternal"``,
    ``"paternal"``, or ``"both"`` (default) — each selected copy carrying
    *from_allele* converts independently at *rate*.  The somatic label
    (slab) never changes.
    """

    side: str

    def __init__(
        self,
        *,
        from_allele: object,
        to_allele: object,
        rate: object,
        side: object = "both",
        filters: Optional[Mapping[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize one allele-level zygote conversion.

        Args:
            from_allele: Source allele (gene) name; locates the locus.
                Must be a non-empty string.
            to_allele: Target allele; must belong to the same locus.
                Must be a non-empty string.
            rate: Conversion probability in ``[0, 1]``.
            side: ``"maternal"`` / ``"paternal"`` / ``"both"`` (default).
            filters: ``current`` / ``maternal`` / ``paternal`` patterns.
            name: Optional display name.

        Raises:
            ValueError: If *rate*, *filters* or *side* are invalid.
            TypeError: If an allele name is not a string.
        """
        if not isinstance(from_allele, str) or not from_allele:  # runtime boundary
            raise TypeError(f"from_allele must be a non-empty string, got {from_allele!r}")
        if not isinstance(to_allele, str) or not to_allele:
            raise TypeError(f"to_allele must be a non-empty string, got {to_allele!r}")
        if not isinstance(side, str) or side not in SIDES:  # runtime boundary
            raise ValueError(
                f"side must be one of {list(SIDES)}, got {side!r}"
            )
        # `side` selects which zygotic copies are eligible to convert; each
        # selected copy later converts independently at `rate`.
        self.from_allele: str = from_allele
        self.to_allele: str = to_allele
        self.side: str = side
        self._init_common(rate, filters, name, ZYGOTE_FILTER_KEYS, "zygote")

    def __repr__(self) -> str:
        """Return a readable rule identity."""
        return (
            f"ZygoteAlleleConversionRule({self.from_allele}->{self.to_allele}, "
            f"rate={self.rate}, side={self.side})"
        )


def replace_allele_in_haploid(
    hg: HaploidGenotype,
    from_allele: str,
    to_allele: str,
) -> Optional[HaploidGenotype]:
    """Return a new ``HaploidGenotype`` with *from_allele* → *to_allele*.

    Scans every gene in *hg*.  When a gene named *from_allele* is found,
    the *to_allele* ``Gene`` registered at the same locus is substituted
    and a new (cached) ``HaploidGenotype`` is constructed.  Entity caching
    makes the result an identity-resolvable object.

    Args:
        hg: The haploid genotype to scan.
        from_allele: Name of the source allele to replace.
        to_allele: Name of the same-locus target allele.

    Returns:
        The converted haploid genotype, or ``None`` when *from_allele* is
        absent or the target allele is not registered at that locus.
    """
    from natal.frontend.genetics import Haplotype
    from natal.frontend.genetics.entities.haplotype import HaploidGenotype

    species = hg.species

    # A locus belongs to exactly one chromosome, so the first matching gene is
    # the unique source copy in this haploid genome.
    for hap_idx, haplotype in enumerate(hg.haplotypes):
        for gene in haplotype.genes:
            if gene.name != from_allele:
                continue

            locus = gene.locus
            # Locate the same-locus target; the compile step already verified it
            # is registered, so a miss here means "not this locus".
            target_gene = None
            for registered in locus.all_entities:
                if registered.name == to_allele:
                    target_gene = registered
                    break

            if target_gene is None:
                # No convertible target at this locus: skip and keep scanning.
                continue

            new_genes = [
                target_gene if g is gene else g
                for g in haplotype.genes
            ]
            new_haplotype = Haplotype(
                chromosome=haplotype.chromosome,
                genes=new_genes,
            )

            new_haplotypes = [
                new_haplotype if i == hap_idx else h
                for i, h in enumerate(hg.haplotypes)
            ]
            # Only the affected haplotype is rebuilt; the rest are reused, and
            # the HaploidGenotype constructor returns a cached instance.
            return HaploidGenotype(species=species, haplotypes=new_haplotypes)

    return None
