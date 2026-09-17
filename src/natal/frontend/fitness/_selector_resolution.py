"""Shared resolution of a fitness selector to ZType indices.

A fitness selector addresses *ZTypes* — a genotype together with its somatic
label — so an ``@slab`` qualifier is part of the selection, not decoration.
Two entry points write fitness tensors (the ``fitness()`` chain and a preset's
``fitness_patch()``) and they must agree: the same selector string has to
select the same ZTypes on both (FRONTEND_REFACTOR_PLAN.md §5.6).  This module
is the resolution both entries call for a labelled selector; a selector
without a label is resolved genotype-by-genotype at each call site.

``_writer.py``'s flat viability/fecundity/zygote path still carries its own
inline copy of the labelled logic.  A 252-case scan (3 species x 3 fields x 28
labelled selectors, preset against chain) found no divergence, and folding
that copy in is part of the §5 migration rather than a behaviour fix.

Private module — not part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple, Union

from natal.frontend.genetics import Genotype, Species
from natal.frontend.registry.index import IndexRegistry

__all__ = ["resolve_selector_ztypes"]

_SelectorT = Union[Genotype, str, Tuple[Union[Genotype, str], ...]]


def resolve_selector_ztypes(
    species: Species,
    registry: IndexRegistry,
    selector: _SelectorT,
    all_genotypes: Iterable[Genotype],
    context: str,
) -> List[int]:
    """Resolve one selector to the ZType indices it addresses.

    A selector carrying an ``@slab`` suffix is resolved through
    ``ZygoteTypePattern``, which is label-aware: only ZTypes whose label
    matches come back, and a label that no ZType carries raises instead of
    being ignored.  A selector without a label keeps the genotype-level
    resolution unchanged — match genotypes, then address every slab of each.

    Args:
        species: Species providing the genotype catalog and the label set.
        registry: Index registry mapping genotypes and labels to ZTypes.
        selector: Selector to resolve (a string, a ``Genotype``, or a tuple of
            those).
        all_genotypes: Candidate genotypes for the unlabelled fallback.
        context: Error-message prefix naming the caller.

    Returns:
        ZType indices in a stable order, without duplicates.

    Raises:
        ValueError: If a labelled selector matches no ZType.
    """
    if isinstance(selector, str) and "@" in selector:
        return _labelled_selector_ztypes(species, registry, selector, context)

    out: List[int] = []
    seen: set[int] = set()
    for genotype in species.resolve_genotype_selectors(
        selector=selector,
        all_genotypes=all_genotypes,
        context=context,
    ):
        for ztype in registry.ztype_indices_for(genotype):
            if ztype not in seen:
                seen.add(ztype)
                out.append(ztype)
    return out


def _labelled_selector_ztypes(
    species: Species,
    registry: IndexRegistry,
    selector: str,
    context: str,
) -> List[int]:
    """Resolve a selector whose ``@slab`` qualifier must take part in matching."""
    from natal.frontend.patterns import ZygoteTypePattern

    ztypes = list(registry.resolve_ztype_indices(ZygoteTypePattern.parse(selector, species)))
    # ``|`` asks for an ordered pair.  On an unordered species the registry
    # holds only the canonical phase, so an ordered spelling can miss matches
    # that its ``::`` form finds; retry with the unordered separator and keep
    # the wider result.  This mirrors the chain path exactly, so adding a label
    # never narrows the genotype match set.
    if species.unordered and "|" in selector and "::" not in selector:
        try:
            promoted = ZygoteTypePattern.parse(selector.replace("|", "::", 1), species)
            promoted_ztypes = list(registry.resolve_ztype_indices(promoted))
            if len(promoted_ztypes) >= len(ztypes):
                ztypes = promoted_ztypes
        except Exception:
            pass
    if not ztypes:
        raise ValueError(
            f"{context}: selector {selector!r} matches no ZType in species "
            f"{species.name!r}."
        )
    return ztypes
