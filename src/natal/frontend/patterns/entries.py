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

import re
from collections.abc import Collection
from typing import Literal, overload

from natal.frontend.genetics import Species

from .elements._base import PatternParseError
from .elements.atom import LabPattern
from .elements.diploid import GenotypePattern, ZygoteTypePattern
from .elements.haploid import GameteTypePattern, HaploidGenomePattern
from .parser import ConversionTarget, GenotypePatternParser

SelectorKind = Literal["genotype", "haploid", "ztype", "gtype"]


def _promote_unordered(species: Species, spec: str, *, ordered: bool) -> str:
    """Rewrite every ordered pair separator for an unordered species.

    The ``\\x00`` placeholder preserves any ``::`` the user already wrote,
    so promotion is idempotent and never turns an explicit unordered
    separator into anything else.  A haploid spelling contains no ``|`` and
    passes through unchanged.
    """
    if ordered or not species.unordered:
        return spec
    return spec.replace("::", "\x00").replace("|", "::").replace("\x00", "::")


def _validate_selector_label(
    species: Species,
    label: LabPattern | None,
    *,
    kind: SelectorKind,
    context: str,
    label_catalog: Collection[str] | None = None,
) -> None:
    """Reject selector labels that are absent from the species catalog.

    ``LabPattern`` deliberately treats a negated unknown label as a useful
    complement.  A selector is a user-facing species query, though, so a
    typo must fail before it can broaden a match to every registered label.
    The same check covers exact labels and every member of a label set.
    """
    if label is None or kind not in ("ztype", "gtype"):
        return
    species_labels = (
        species.somatic_labels if kind == "ztype" else species.gamete_labels
    ) or ["default"]
    # A runtime registry can be compressed, so its labels are only an
    # additional binding context.  Never let that subset hide labels declared
    # by the species itself.
    labels = list(dict.fromkeys((*species_labels, *(label_catalog or ()))))
    named = label.lab_set or ({label.lab} if label.lab is not None else set())
    unknown = set(named) - set(labels)
    if unknown:
        raise ValueError(
            f"{context}: selector names unknown {kind} labels "
            f"{sorted(unknown)!r}; available labels: {list(labels)!r}"
        )


def _validate_selector_alleles(
    species: Species, content: str, *, context: str
) -> None:
    """Reject allele names absent from the species catalog when requested."""
    for token in sorted(set(re.findall(r"[A-Za-z0-9_]+", content))):
        if species.get_gene(token) is None:
            raise ValueError(
                f"{context}: selector pattern names unknown allele {token!r}; "
                "every allele in a selector pattern must be registered in the species"
            )


@overload
def parse_selector(
    spec: str, *, species: Species, context: str = ..., ordered: bool = ...,
    validate_alleles: bool = ...,
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: ZygoteTypePattern, *, species: Species, context: str = ...,
    label_catalog: Collection[str] | None = ...,
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: ZygoteTypePattern, *, species: Species,
    kind: Literal["ztype"], context: str = ...,
    ordered: bool = ..., validate_alleles: bool = ...,
    label_catalog: Collection[str] | None = ...,
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: str | ZygoteTypePattern, *, species: Species,
    kind: Literal["ztype"], context: str = ..., ordered: bool = ...,
    validate_alleles: bool = ..., label_catalog: Collection[str] | None = ...,
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["ztype"], context: str = ...,
    ordered: bool = ..., validate_alleles: bool = ...,
) -> ZygoteTypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["genotype"], context: str = ...,
    ordered: bool = ..., validate_alleles: bool = ...,
) -> GenotypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["gtype"], context: str = ...,
    ordered: bool = ..., validate_alleles: bool = ...,
) -> GameteTypePattern: ...


@overload
def parse_selector(
    spec: str, *, species: Species, kind: Literal["haploid"], context: str = ...,
    ordered: bool = ..., validate_alleles: bool = ...,
) -> HaploidGenomePattern: ...


def parse_selector(
    spec: object,
    *,
    species: Species,
    kind: SelectorKind = "ztype",
    context: str = "selector",
    ordered: bool = False,
    validate_alleles: bool = False,
    label_catalog: Collection[str] | None = None,
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
        ordered: Keep ``|`` strictly ordered even for an unordered species.
            Species content-only helpers set this explicitly to preserve their
            historical contract; matching selectors use the default promotion.
        validate_alleles: Also reject allele names absent from the species
            catalog. Conversion-rule filters enable this stricter boundary;
            ordinary selectors retain their no-match behavior for unknown
            content.
        label_catalog: Optional labels registered by the binding registry.
            This augments the species catalog for manually assembled registries
            while keeping typo rejection at the unified selector entry.

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
    if isinstance(spec, ZygoteTypePattern):
        _validate_selector_label(
            species, spec.slab, kind="ztype", context=context,
            label_catalog=label_catalog,
        )
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"{context} selector must be a string, got {type(spec).__name__}")
    text = _promote_unordered(species, spec.strip(), ordered=ordered)
    parser = GenotypePatternParser(species)
    if kind == "ztype":
        if "@" in text and text.count("@") == 1:
            base_text, label_text = text.rsplit("@", 1)
            if not base_text.strip() or not label_text.strip():
                raise PatternParseError("empty genotype or label")
        content, slab = GenotypePatternParser.split_label_suffix(text)
        _validate_selector_label(
            species, slab, kind=kind, context=context, label_catalog=label_catalog
        )
        if validate_alleles:
            _validate_selector_alleles(species, content, context=context)
        return ZygoteTypePattern(
            parser._parse(content),  # pyright: ignore[reportPrivateUsage]  # this entry IS the public path
            slab,
            source_text=content.strip(),
        )
    if kind == "genotype":
        if validate_alleles:
            _validate_selector_alleles(species, text, context=context)
        return parser._parse(text)  # pyright: ignore[reportPrivateUsage]  # this entry IS the public path
    if kind == "gtype":
        if "@" in text and text.count("@") == 1:
            base_text, label_text = text.rsplit("@", 1)
            if not base_text.strip() or not label_text.strip():
                raise PatternParseError("empty genotype or label")
        result = parser._parse_gamete(text)  # pyright: ignore[reportPrivateUsage]  # this entry IS the public path
        _validate_selector_label(
            species, result.glab, kind=kind, context=context,
            label_catalog=label_catalog,
        )
        if validate_alleles:
            content, _ = GenotypePatternParser.split_label_suffix(text)
            _validate_selector_alleles(species, content, context=context)
        return result
    if kind == "haploid":
        if validate_alleles:
            _validate_selector_alleles(species, text, context=context)
        return parser._parse_haploid(text)  # pyright: ignore[reportPrivateUsage]  # this entry IS the public path
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
        target: Unvalidated runtime input; a target pattern string such as
            ``"B|b@marked"`` or ``"*@*"``, or a structured pattern returned
            by :func:`parse_selector`. Gamete targets use ``"A1; B1@glab"``.
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
        TypeError: If *target* is neither a string nor a structured pattern
            compatible with the requested gamete or zygote kind.
        PatternParseError: If syntax is invalid or a required part is
            absent.
        ValueError: If a structured pattern lacks its source text, or
            *validate* is true and a form can select multiple alternatives
            instead of specifying a change.
    """
    if isinstance(target, ZygoteTypePattern):
        if haploid:
            raise TypeError("haploid conversion target requires a GameteTypePattern")
        if require_label and target.slab is None:
            raise PatternParseError(
                "conversion target must provide both genotype and label parts"
            )
        if target.source_text is None:
            raise ValueError(
                "structured zygote target does not retain its source pattern text"
            )
        label = target.slab or LabPattern()
        label_text = label.lab if label.lab is not None else "*"
        parsed = ConversionTarget(
            target.source_text, label_text, target.genotype, label
        )
        if validate:
            parsed.validate(species)
        return parsed
    if isinstance(target, GameteTypePattern):
        if not haploid:
            raise TypeError("zygote conversion target requires a ZygoteTypePattern")
        if require_label and target.glab is None:
            raise PatternParseError(
                "conversion target must provide both genotype and label parts"
            )
        if target.source_text is None:
            raise ValueError(
                "structured gamete target does not retain its source pattern text"
            )
        label = target.glab or LabPattern()
        label_text = label.lab if label.lab is not None else "*"
        parsed = ConversionTarget(target.source_text, label_text, target.genome, label)
        if validate:
            parsed.validate(species)
        return parsed

    parser = GenotypePatternParser(species)
    parsed = parser._parse_conversion_target(  # pyright: ignore[reportPrivateUsage]  # this entry IS the public path
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
