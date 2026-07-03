"""
Atom-level pattern elements: AllelePattern, WildcardPattern, SetPattern,
LabPattern, LocusPattern.
"""

from __future__ import annotations

from typing import Optional, Set

from natal.genetics import Gene

from ._base import PatternElement, PatternParseError


class AllelePattern(PatternElement):
    """Exact match for a single allele name."""

    def __init__(self, allele_name: str):
        self.allele_name = allele_name

    def matches(self, gene: Optional[Gene]) -> bool:
        if gene is None:
            return False
        return gene.name == self.allele_name

    def __repr__(self) -> str:
        return f"AllelePattern({self.allele_name})"


class WildcardPattern(PatternElement):
    """Wildcard (*) - matches any allele."""

    def matches(self, gene: Optional[Gene]) -> bool:
        return gene is not None

    def __repr__(self) -> str:
        return "WildcardPattern(*)"


class SetPattern(PatternElement):
    """Set pattern - matches alleles in a set, with optional negation."""

    def __init__(self, alleles: Set[str], negate: bool = False):
        """Initialize a set pattern.

        Args:
            alleles: Set of allele names to match.
            negate: If True, match alleles NOT in this set.
        """
        self.alleles = alleles
        self.negate = negate

    def matches(self, gene: Optional[Gene]) -> bool:
        if gene is None:
            return False
        result = gene.name in self.alleles
        return (not result) if self.negate else result

    def __repr__(self) -> str:
        prefix = "!" if self.negate else ""
        return f"SetPattern({prefix}{{{', '.join(sorted(self.alleles))}}})"


class LocusPattern:
    """Pattern for a single locus (two homologous chromosomes)."""

    def __init__(
        self,
        maternal_pattern: PatternElement,
        paternal_pattern: PatternElement,
        unordered: bool = False
    ):
        """Initialize a locus pattern.

        Args:
            maternal_pattern: PatternElement for maternal allele.
            paternal_pattern: PatternElement for paternal allele.
            unordered: If True, use :: ordering (match either maternal|paternal or paternal|maternal).
        """
        self.maternal_pattern = maternal_pattern
        self.paternal_pattern = paternal_pattern
        self.unordered = unordered

    def matches(self, mat_gene: Optional[Gene], pat_gene: Optional[Gene]) -> bool:
        """Check if a pair of alleles matches this locus pattern.

        Args:
            mat_gene: Maternal allele.
            pat_gene: Paternal allele.

        Returns:
            True if the allele pair matches.
        """
        if self.unordered:
            # Try both orderings
            match_straight = (
                self.maternal_pattern.matches(mat_gene) and
                self.paternal_pattern.matches(pat_gene)
            )
            match_reversed = (
                self.maternal_pattern.matches(pat_gene) and
                self.paternal_pattern.matches(mat_gene)
            )
            return match_straight or match_reversed
        else:
            # Strict ordering
            return (
                self.maternal_pattern.matches(mat_gene) and
                self.paternal_pattern.matches(pat_gene)
            )

    def __repr__(self) -> str:
        sep = "::" if self.unordered else "/"
        return f"{self.maternal_pattern}{sep}{self.paternal_pattern}"


class LabPattern:
    """Pattern for matching gamete / somatic label names.

    Supports the same syntax as allele patterns:
      - ``@cas9_high`` — exact match
      - ``@!cas9_high`` — any label except "cas9_high"
      - ``@{cas9_high,cas9_low}`` — any label in the set
      - ``@!{cas9_high,cas9_low}`` — any label NOT in the set

    When *lab* and *lab_set* are both ``None`` the pattern matches any label
    (equivalent to omitting the ``@`` suffix entirely).
    """

    def __init__(
        self,
        lab: Optional[str] = None,
        negate: bool = False,
        lab_set: Optional[Set[str]] = None,
    ):
        self.lab = lab
        self.negate = negate
        self.lab_set = lab_set

    def matches(self, value: str) -> bool:
        """Return True if *value* satisfies this pattern."""
        if self.lab_set is not None:
            result = value in self.lab_set
            return not result if self.negate else result
        if self.lab is not None:
            result = value == self.lab
            return not result if self.negate else result
        return True  # wildcard — matches anything

    def __bool__(self) -> bool:
        """False when this is a pure wildcard (no constraint).

        Prefer :meth:`is_wildcard` for readability in conditional checks.
        """
        return self.lab is not None or self.lab_set is not None

    def is_wildcard(self) -> bool:
        """True if this pattern matches any label (no constraint)."""
        return self.lab is None and self.lab_set is None

    def __repr__(self) -> str:
        if self.lab_set is not None:
            inner = "{" + ",".join(sorted(self.lab_set)) + "}"
        elif self.lab is not None:
            inner = self.lab
        else:
            return "LabPattern(*)"
        prefix = "!" if self.negate else ""
        return f"LabPattern({prefix}{inner})"

    @staticmethod
    def parse(lab_str: str) -> LabPattern:
        """Parse a ``@lab`` suffix string into a ``LabPattern``.

        Returns a wildcard (matches-everything) pattern for ``"*"``
        or the empty string.
        """
        from natal.utils.helpers import validate_name

        s = lab_str.strip()
        if not s or s == "*":
            return LabPattern()

        negate = False
        if s.startswith("!"):
            negate = True
            s = s[1:].strip()

        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1].strip()
            if not inner:
                raise PatternParseError("Empty lab set {}")
            names = {n.strip() for n in inner.split(",")}
            for name in names:
                if not validate_name(name):
                    raise PatternParseError(
                        f"Invalid lab name {name!r} in set. "
                        f"Lab names must match [A-Za-z0-9_]+."
                    )
            return LabPattern(lab_set=names, negate=negate)

        # Single name or comma-separated (set without braces)
        if "," in s:
            names = {n.strip() for n in s.split(",")}
            for name in names:
                if not validate_name(name):
                    raise PatternParseError(
                        f"Invalid lab name {name!r}. "
                        f"Lab names must match [A-Za-z0-9_]+."
                    )
            return LabPattern(lab_set=names, negate=negate)

        if not validate_name(s):
            raise PatternParseError(
                f"Invalid lab name {s!r}. "
                f"Lab names must match [A-Za-z0-9_]+."
            )
        return LabPattern(lab=s, negate=negate)
