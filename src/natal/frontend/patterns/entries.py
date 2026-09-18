"""The two semantic pattern entries: selection and conversion targeting.

FRONTEND_REFACTOR_PLAN.md §5.1: business code reaches the pattern grammar
through exactly one entry per semantic — :func:`parse_selector` for
matching, :func:`parse_target` for keep-or-replace conversion.  Both share
the grammar's own analysis (chromosome groups, haplotypes, loci, alleles,
wildcards, sets/negation, and the ``@label`` suffix); they differ only in
how an omitted or wildcard part is interpreted — a selector leaves it
unconstrained, a target retains the source's part.

Unordered species (the finalized promotion rule): a selector written with
ordered ``|`` separators is promoted to ``::`` — every separator, with any
``::`` the user already wrote preserved — before parsing.  One spelling
therefore selects the same individuals through fitness, rules, conversion
filters, observation, and hooks.  Content-only parsing (the ``Species``
``parse_*``/``enumerate_*``/``filter_*`` helpers) does **not** promote:
``|`` there remains the strict-order grammar documented in the pattern
guide.  Targets are never promoted — a replacement must say which side it
replaces, and ``ConversionTarget.validate`` rejects the ambiguous forms.
"""

from __future__ import annotations

from typing import Literal, overload

from natal.frontend.genetics import Species

from .elements.diploid import GenotypePattern, ZygoteTypePattern
from .elements.haploid import GameteTypePattern, HaploidGenomePattern
from .parser import ConversionTarget, GenotypePatternParser

SelectorKind = Literal["genotype", "haploid", "ztype", "gtype"]


def _promote_unordered(species: Species, spec: str) -> str:
    """Rewrite every ordered pair separator for an unordered species.

    The ``\\x00`` placeholder preserves any ``::`` the user already wrote,
    so promotion is idempotent and never turns an explicit unordered
    separator into anything else.  A haploid spelling contains no ``|`` and
    passes through unchanged.
    """
    if not species.unordered:
        return spec
    return spec.replace("::", "\x00").replace("|", "::").replace("\x00", "::")


@overload
def parse_selector(
    spec: str, *, species: Species, context: str = ...
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["ztype"], context: str = ...
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["genotype"], context: str = ...
) -> GenotypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["gtype"], context: str = ...
) -> GameteTypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["haploid"], context: str = ...
) -> HaploidGenomePattern: ...


def parse_selector(
    spec: object,
    *,
    species: Species,
    kind: SelectorKind = "ztype",
    context: str = "selector",
) -> object:
    """Parse one selector expression with the single matching semantic.

    The kind fixes what the expression may carry and what it matches:

    - ``"ztype"`` (default): a genotype pattern plus an optional ``@slab``
      suffix, matching whole zygote types.
    - ``"genotype"``: genetic content only; an ``@label`` suffix is
      rejected because a genotype has no label to match.
    - ``"gtype"``: a haploid genome pattern plus an optional ``@glab``
      suffix, matching whole gamete types.
    - ``"haploid"``: haploid genetic content only; ``@label`` is rejected.

    On an unordered species every ``|`` is promoted to ``::`` first (see
    the module docstring), so the same spelling matches identically from
    every caller.  Binding the returned pattern to concrete indices is the
    registry's stage (``IndexRegistry.resolve_ztype_indices`` and friends),
    not a second parse.

    Args:
        spec: Selector expression, e.g. ``"A|a@infected"``, ``"*|A; *|B"``,
            or ``"A1/B1; C1@deposited"``.
        species: Species whose chromosome groups and unordered flag define
            the grammar's context.
        kind: What the selector matches; selects the return type.
        context: Label identifying the caller in error messages.

    Returns:
        A pattern object of the kind requested: ``ZygoteTypePattern``,
        ``GenotypePattern``, ``GameteTypePattern``, or
        ``HaploidGenomePattern``.

    Raises:
        TypeError: If *spec* is not a string.
        ValueError: If *kind* is not one of the four known kinds.
        PatternParseError: If the expression is malformed, uses a label
            where the kind carries none, or names more chromosome groups
            than the species has.
    """
    if not isinstance(spec, str):
        raise TypeError(f"{context} selector must be a string, got {type(spec).__name__}")
    text = _promote_unordered(species, spec.strip())
    parser = GenotypePatternParser(species)
    if kind == "ztype":
        content, slab = GenotypePatternParser.split_label_suffix(text)
        return ZygoteTypePattern(parser.parse(content), slab)
    if kind == "genotype":
        return parser.parse(text)
    if kind == "gtype":
        return parser.parse_haplotype_pattern(text)
    if kind == "haploid":
        return parser.parse_haploid_genome_pattern(text)
    raise ValueError(
        f"{context} selector kind {kind!r} is unknown; expected one of "
        "'genotype', 'haploid', 'ztype', 'gtype'"
    )


def parse_target(
    target: object,
    *,
    species: Species,
    stage: str = "conversion",
    haploid: bool = False,
    require_label: bool = False,
    validate: bool = False,
) -> ConversionTarget:
    """Parse one conversion target with the single keep-or-replace semantic.

    An omitted or ``*`` part retains the corresponding part of each source
    entity; a concrete value replaces it.  Each source must resolve to
    exactly one legal destination — ambiguous forms (sets, negations,
    unordered pairs, bracketed locus pairs) are rejected rather than
    guessed.  Binding happens in stages, mirroring :func:`parse_selector`:
    this entry is the syntax stage, ``ConversionTarget.validate`` the
    species-bound catalog stage, and ``apply_zygote``/``apply_gamete`` the
    per-source layout stage.

    Args:
        target: Unvalidated runtime input; must be a target pattern string
            such as ``"B|b@marked"`` or ``"*@*"``, or ``"A1; B1@glab"`` for
            a gamete target.
        species: Species whose chromosome groups define the grammar.
        stage: Context included in validation errors.
        haploid: Whether the target describes a gamete rather than a
            zygote.
        require_label: Require the explicit legacy ``genotype@label``
            form; otherwise an omitted label retains the source label.
        validate: Also run the species-bound ambiguity validation now
            instead of leaving it to the caller's binding stage.

    Returns:
        The structured target applied separately to each source entity.

    Raises:
        TypeError: If *target* is not a string.
        PatternParseError: If syntax is invalid or a required part is
            absent.
        ValueError: If *validate* is true and a form can select multiple
            alternatives instead of specifying a change.
    """
    parser = GenotypePatternParser(species)
    parsed = parser.parse_conversion_target(
        target, stage=stage, haploid=haploid, require_label=require_label
    )
    if validate:
        parsed.validate(species)
    return parsed


__all__ = [
    "SelectorKind",
    "parse_selector",
    "parse_target",
]
