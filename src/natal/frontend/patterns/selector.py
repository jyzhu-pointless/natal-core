"""Zygote-type resolution for selector strings.

Provides :func:`resolve_zygote_type`, which funnels one selector string
through the unified selector entry and binds it to registry ZType indices.
"""

from __future__ import annotations

from natal.frontend.genetics import Species
from natal.frontend.registry.index import IndexRegistry


def resolve_zygote_type(
    spec: str,
    species: Species,
    index_registry: IndexRegistry,
) -> list[int]:
    """Resolve a genotype string to ZType indices, with species-appropriate matching.

    For unordered species, the selector entry promotes ``|`` to ``::`` so
    that ``"A|a"`` matches both ordered and unordered (canonicalized)
    registrations — the same promotion every other selector caller gets.

    For ordered species (e.g. sex chromosomes), ``|`` is treated strictly —
    ``"a|A"`` and ``"A|a"`` are distinct genotypes and will each only match
    their exact ordering.

    Does NOT perform the reversed-maternal/paternal fallback (that would be
    a bug for ordered species).

    Args:
        spec: Genotype selector string (e.g. ``"A|A"``, ``"Drive|WT"``,
            ``"*"``, ``"A@exposed"``).
        species: Species for genotype-resolution context.
        index_registry: Registry for ZType index resolution.

    Returns:
        List of matching ZType indices (may be empty if nothing matches).
    """
    from .entries import parse_selector

    pattern = parse_selector(spec, species=species, kind="ztype", context="zygote type")
    return index_registry.resolve_ztype_indices(pattern)
