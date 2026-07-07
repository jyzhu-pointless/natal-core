"""Atom-level pattern elements: AllelePattern, WildcardPattern, SetPattern,
LabPattern, LocusPattern.

Provides the primitive pattern matching elements that operate at the
individual-allele level, plus label pattern matching (:class:`LabPattern`)
and locus-pair pattern matching (:class:`LocusPattern`).
"""

from __future__ import annotations

from typing import Optional, Set

from natal.genetics import Gene

from ._base import PatternElement, PatternParseError


class AllelePattern(PatternElement):
    """Exact match for a single allele name.

    Attributes:
        allele_name (str): The exact allele name this pattern matches.
    """

    def __init__(self, allele_name: str):
        """Initialize an AllelePattern.

        Args:
            allele_name: The exact allele name to match.
        """
        self.allele_name = allele_name

    def matches(self, gene: Optional[Gene]) -> bool:
        """Check if a gene matches this allele name.

        Args:
            gene: The Gene object to match, or ``None``.

        Returns:
            True if *gene* is not ``None`` and its name equals
            ``self.allele_name``.
        """
        if gene is None:
            return False
        return gene.name == self.allele_name

    def __repr__(self) -> str:
        """Return a string representation of this pattern."""
        return f"AllelePattern({self.allele_name})"


class WildcardPattern(PatternElement):
    """Wildcard pattern - matches any allele.

    The wildcard (``*``) matches any non-``None`` gene.
    """

    def matches(self, gene: Optional[Gene]) -> bool:
        """Check if a gene matches this wildcard.

        Args:
            gene: The Gene object to match, or ``None``.

        Returns:
            True if *gene* is not ``None``.
        """
        return gene is not None

    def __repr__(self) -> str:
        """Return a string representation of this pattern."""
        return "WildcardPattern(*)"


class SetPattern(PatternElement):
    """Set pattern - matches alleles in a set, with optional negation.

    Supports positive matching (allele is in the set), negated matching
    (allele is NOT in the set), and multi-allele set syntax like
    ``{A,B,C}`` and ``!{A,B}``.

    Attributes:
        alleles (Set[str]): Set of allele names.
        negate (bool): If True, match alleles NOT in the set.
    """

    def __init__(self, alleles: Set[str], negate: bool = False):
        """Initialize a set pattern.

        Args:
            alleles: Set of allele names to match.
            negate: If True, match alleles NOT in this set.
        """
        self.alleles = alleles
        self.negate = negate

    def matches(self, gene: Optional[Gene]) -> bool:
        """Check if a gene matches this set pattern.

        Args:
            gene: The Gene object to match, or ``None``.

        Returns:
            True if *gene* is not ``None`` and satisfies the set constraint.
        """
        if gene is None:
            return False
        result = gene.name in self.alleles
        return (not result) if self.negate else result

    def __repr__(self) -> str:
        """Return a string representation of this pattern."""
        prefix = "!" if self.negate else ""
        return f"SetPattern({prefix}{{{', '.join(sorted(self.alleles))}}})"


class LocusPattern:
    """Pattern for a single locus on two homologous chromosomes.

    Matches a pair of alleles (maternal and paternal) at one locus.
    Supports ordered (``|``) and unordered (``::``) matching.

    Attributes:
        maternal_pattern (PatternElement): Pattern for the maternal allele.
        paternal_pattern (PatternElement): Pattern for the paternal allele.
        unordered (bool): If True, maternal/paternal order is ignored.
    """

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
        """Return a string representation of this locus pattern."""
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

    Attributes:
        lab (Optional[str]): Exact label to match, or ``None``.
        negate (bool): If True, negate the matching condition.
        lab_set (Optional[Set[str]]): Set of labels for set matching.
    """

    def __init__(
        self,
        lab: Optional[str] = None,
        negate: bool = False,
        lab_set: Optional[Set[str]] = None,
    ):
        """Initialize a LabPattern.

        Args:
            lab: Exact label name to match, or ``None`` for wildcard.
            negate: If True, match the complement of the constraint.
            lab_set: Set of label names for set matching.
        """
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

        Returns:
            True if a constraint (lab or lab_set) is set.
        """
        return self.lab is not None or self.lab_set is not None

    def is_wildcard(self) -> bool:
        """True if this pattern matches any label (no constraint)."""
        return self.lab is None and self.lab_set is None

    def __repr__(self) -> str:
        """Return a string representation of this label pattern."""
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
