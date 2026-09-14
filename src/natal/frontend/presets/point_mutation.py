"""Point-mutation preset.

Public module — provides PointMutation for spontaneous germline allele
conversion with competing multi-target semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, cast

from natal.frontend.genetics import Gene
from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.utils.helpers import resolve_sex_label
from natal.frontend.utils.types import Sex

from ._base import GeneticPreset
from ._fitness import make_fitness_patch_given_allele_scaling
from ._types import (
    AlleleScalingMode,
    AlleleSpecifier,
    FecundityScalingConfig,
    PresetFitnessPatch,
    SexSpecificRates,
    SexualSelectionScalingConfig,
    ViabilityScalingConfig,
    ZygoteViabilityScalingConfig,
    coerce_sex_specifier,
)

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import RecipeHost

# How declared rates are interpreted:
#
# - "strict": each rate is the target's effective probability, so the sum must
#   stay at or below 1 (a source allele cannot mutate into two targets at once).
# - "proportional": the rates are relative proportions and are scaled to sum to 1.
RateMode = Literal["strict", "proportional"]

_RATE_MODES: Tuple[RateMode, ...] = ("strict", "proportional")


def _correct_competing(rates: Sequence[float], rate_mode: str, context: str) -> List[float]:
    """Adjust declared rates so every target receives its declared share.

    The conversion rulesets cascade their rules in declaration order and hand
    each rule only the source mass the previous rule left unconverted.  Passing
    rule ``k`` its raw declared rate ``r_k`` would therefore give it the
    effective share ``r_k * (1 - sum(r_i, i < k))``.  Dividing by the remaining
    mass restores the declared share::

        r'_k = r_k / (1 - sum(r_i, i < k))

    Args:
        rates: Declared rates, one per competing target (floats).
        rate_mode: ``"strict"`` or ``"proportional"``.
        context: Error-message prefix identifying the preset and parameter.

    Returns:
        The adjusted rates in declaration order.

    Raises:
        ValueError: If *rate_mode* is unknown, a rate is non-finite or
            negative, the rates exceed the allowed total in ``"strict"``
            mode, or they leave no unconverted source mass for a positive
            rate.
    """
    if rate_mode not in _RATE_MODES:
        raise ValueError(
            f"{context}: rate_mode must be one of {list(_RATE_MODES)}, "
            f"got {rate_mode!r}"
        )
    resolved = [float(rate) for rate in rates]
    # Both modes require finite, non-negative declarations: "strict" rates are
    # probabilities, "proportional" rates are relative weights and may exceed 1
    # before normalization.  Rejecting here keeps NaN (which no comparison
    # below would catch) and negative weights from silently becoming rates.
    for rate in resolved:
        if not math.isfinite(rate):
            raise ValueError(f"{context}: a declared rate must be finite, got {rate!r}")
        if rate < 0.0:
            raise ValueError(
                f"{context}: a declared rate must not be negative, got {rate!r}"
            )
    total = sum(resolved)
    if rate_mode == "proportional":
        # Proportions are always normalized, so ratio-style input such as
        # (2, 3, 5) expresses the same model as (0.2, 0.3, 0.5).
        if total > 0.0:
            resolved = [rate / total for rate in resolved]
    elif total > 1.0:
        raise ValueError(
            f"{context}: declared rates sum to {total}, which exceeds 1; "
            "lower the declaration or use rate_mode='proportional' to treat "
            "the rates as proportions"
        )

    adjusted: List[float] = []
    remaining = 1.0
    for rate in resolved:
        if rate <= 0.0:
            # A zero-rate target converts nothing and consumes no source mass.
            adjusted.append(0.0)
            continue
        if remaining <= 0.0:
            # Reachable when the declared rates sum to just above 1 by value
            # but round down to exactly 1 in floating point.
            raise ValueError(
                f"{context}: declared rates leave no unconverted source mass; "
                "their sum must stay below 1"
            )
        # Validated rates cannot exceed the remaining mass mathematically; the
        # clamp only absorbs floating-point dust from the division.
        adjusted.append(min(1.0, rate / remaining))
        remaining -= rate
    return adjusted


def _allele_name(allele: object, field: str, preset_name: str) -> str:
    """Resolve one allele input to its name.

    Raises:
        TypeError: If the input is neither a Gene nor a non-empty string.
    """
    if isinstance(allele, Gene):
        return allele.name
    if isinstance(allele, str) and allele:
        return allele
    raise TypeError(
        f"PointMutation {preset_name!r}: {field} must be a Gene or a "
        f"non-empty string, got {allele!r}"
    )


def _as_number(value: object, field: str, preset_name: str) -> float:
    """Validate a runtime value as a rate number.

    ``reconfigure_preset`` can write any value through ``setattr``, so rate
    values are validated structurally rather than trusted from the annotation.

    Raises:
        ValueError: If *value* is not a number.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"PointMutation {preset_name!r}: {field} must be a number, got "
            f"{value!r}"
        )
    return float(value)


def _as_number_pair(value: object) -> Optional[Tuple[float, float]]:
    """Return ``(female, male)`` when *value* is a two-element numeric pair."""
    if not isinstance(value, (tuple, list)):
        return None
    items = cast("Sequence[object]", value)  # runtime boundary: elements untyped
    if len(items) != 2:
        return None
    first, second = items[0], items[1]
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return None
    return (float(first), float(second))


def _rate_declarations(raw: object, preset_name: str) -> List[object]:
    """Split a runtime ``mutation_rates`` value into per-target declarations.

    A bare number or per-sex mapping is the single-target form written back by
    ``reconfigure_preset``; every other accepted value must be a sequence with
    one declaration per target.

    Raises:
        ValueError: If *raw* is not a number, mapping, or sequence.
    """
    if isinstance(raw, (int, float, Mapping)):
        return [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"PointMutation {preset_name!r}: mutation_rates must be a sequence "
            f"with one rate declaration per target, got {raw!r}"
        )
    return list(cast("Sequence[object]", raw))  # runtime boundary: elements untyped


def _resolve_rate_pair(declaration: object, preset_name: str) -> Tuple[float, float]:
    """Normalize one rate declaration into a ``(female, male)`` pair.

    Args:
        declaration: A number (both sexes), a ``(female, male)`` pair, or a
            per-sex mapping whose keys name female/male (``Sex``, ``0``/``1``,
            ``"female"``/``"f"``, ``"male"``/``"m"``).
        preset_name: Preset name used in error messages.

    Returns:
        The resolved pair; a sex missing from a mapping means ``0.0``.

    Raises:
        ValueError: If the declaration has none of the accepted shapes, names
            an unknown sex, or holds a non-numeric rate.
    """
    if isinstance(declaration, (int, float)):
        return (float(declaration), float(declaration))
    pair = _as_number_pair(declaration)
    if pair is not None:
        return pair
    if isinstance(declaration, Mapping):
        mapping = cast("Mapping[object, object]", declaration)  # runtime boundary
        rates: Dict[int, float] = {}
        for key, value in mapping.items():
            try:
                # coerce first: resolve_sex_label guards its input with an
                # assert, which would leak AssertionError (or, under -O, an
                # AttributeError) instead of the documented ValueError.
                sex_index = resolve_sex_label(coerce_sex_specifier(key))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"PointMutation {preset_name!r}: a per-sex rate key must "
                    f"name female or male (or 0/1), got {key!r}"
                ) from exc
            rates[sex_index] = _as_number(value, "a per-sex rate", preset_name)
        return (rates.get(0, 0.0), rates.get(1, 0.0))
    raise ValueError(
        f"PointMutation {preset_name!r}: a mutation rate must be a number, a "
        f"(female, male) pair, or a per-sex mapping, got {declaration!r}"
    )


class PointMutation(GeneticPreset):
    """Spontaneous point mutation of one source allele into one or more targets.

    Every gamete carrying *source_allele* converts to a target allele at the
    declared rate, independently of the parent's genotype (point mutation is
    spontaneous, not drive-induced).  With several targets the conversions
    compete: they are mutually exclusive outcomes of the same event, so each
    target keeps the declared rate instead of losing mass to the targets
    declared before it.  The preset compensates for the ruleset's sequential
    cascade internally, so the declared rates are the effective rates.

    Only the germline (gamete-stage) channel is implemented: the mutation
    happens while gametes are produced, before fertilization.  An embryonic
    (zygote-stage) channel is deliberately deferred — see TODO.md item #14.

    Attributes:
        mutation_rates (Tuple[Tuple[float, float], ...]): Declared
            ``(female_rate, male_rate)`` pair per target, in declaration
            order.  This is the canonical attribute for
            ``reconfigure_preset``; a bare number or per-sex mapping is also
            accepted for the single-target form.
        rate_mode (RateMode): ``"strict"`` (rates are probabilities summing to
            at most 1) or ``"proportional"`` (rates are proportions, scaled to
            sum to 1).

    Examples:
        >>> mutation = PointMutation(
        ...     "A2B", source_allele="A", target_allele="B", mutation_rate=1e-5
        ... )
        >>> multi = PointMutation(
        ...     "MultiMut",
        ...     source_allele="A",
        ...     target_alleles=["B", "C", "D"],
        ...     mutation_rates=[1e-7, 5e-6, 1e-5],
        ... )
        >>> population.apply_preset(mutation)
    """

    def __init__(
        self,
        name: str,
        source_allele: AlleleSpecifier,
        target_allele: Optional[AlleleSpecifier] = None,
        mutation_rate: Optional[SexSpecificRates] = None,
        target_alleles: Optional[Sequence[AlleleSpecifier]] = None,
        mutation_rates: Optional[Sequence[SexSpecificRates]] = None,
        rate_mode: RateMode = "strict",
        viability_scaling: ViabilityScalingConfig = 1.0,
        fecundity_scaling: FecundityScalingConfig = 1.0,
        sexual_selection_scaling: SexualSelectionScalingConfig = 1.0,
        zygote_viability_scaling: ZygoteViabilityScalingConfig = 1.0,
        viability_mode: AlleleScalingMode = "multiplicative",
        fecundity_mode: AlleleScalingMode = "multiplicative",
        sexual_selection_mode: AlleleScalingMode = "multiplicative",
        zygote_viability_mode: AlleleScalingMode = "multiplicative",
        species: Optional[Any] = None,
        priority: int = 0,
    ) -> None:
        """Initialize a point-mutation preset.

        Args:
            name: Name of the preset.
            source_allele: The allele that mutates (Gene or name).
            target_allele: The single target allele (Gene or name).  Mutually
                exclusive with *target_alleles*.
            mutation_rate: Germline conversion rate for the single-target
                form: a float (both sexes), a ``(female, male)`` pair, or a
                per-sex mapping with keys ``Sex``/``"female"``/``"male"``
                (also ``"f"``/``"m"``).  Mutually exclusive with
                *mutation_rates*; a missing sex key means no conversion for
                that sex.
            target_alleles: The target alleles (Genes or names), in
                declaration order.  Mutually exclusive with *target_allele*.
            mutation_rates: One rate declaration per target, each accepting
                the same shapes as *mutation_rate*.  Mutually exclusive with
                *mutation_rate*.
            rate_mode: How declared rates are read — ``"strict"`` requires
                their sum to stay at or below 1 (default), while
                ``"proportional"`` treats them as proportions and scales them
                to sum to 1.
            viability_scaling: Viability scaling applied to every target
                allele (default neutral).
            fecundity_scaling: Fecundity scaling applied to every target
                allele (default neutral).
            sexual_selection_scaling: Sexual-selection scaling applied to
                every target allele (default neutral).
            zygote_viability_scaling: Zygote viability scaling applied to
                every target allele (default neutral).
            viability_mode: Scaling mode for viability.
            fecundity_mode: Scaling mode for fecundity.
            sexual_selection_mode: Scaling mode for scalar sexual-selection values.
            zygote_viability_mode: Scaling mode for zygote viability.
            species: Optional species to bind at construction.
            priority: Execution order — lower values apply first.

        Raises:
            TypeError: If an allele input is not a Gene or a non-empty string.
            ValueError: If the target/rate declaration forms are mixed or
                mismatched, a target repeats or equals the source allele, or
                the declared rates are invalid under *rate_mode*.
        """
        # Bind name/species/priority first so validation errors can name the preset.
        super().__init__(name=name, species=species, priority=priority)

        self._str_source_allele = _allele_name(source_allele, "source_allele", self.name)

        # The two declaration forms are mutually exclusive; each accepts either
        # a single target or an explicit list, but not both at once.
        if target_allele is not None and target_alleles is not None:
            raise ValueError(
                f"PointMutation {self.name!r}: declare either target_allele or "
                "target_alleles, not both"
            )
        targets: Tuple[str, ...]
        if target_allele is not None:
            targets = (_allele_name(target_allele, "target_allele", self.name),)
        elif target_alleles is not None:
            targets = tuple(
                _allele_name(target, "target_alleles", self.name)
                for target in target_alleles
            )
            if not targets:
                raise ValueError(
                    f"PointMutation {self.name!r}: target_alleles must name at "
                    "least one target allele"
                )
        else:
            raise ValueError(
                f"PointMutation {self.name!r}: declare target_allele (single "
                "target) or target_alleles (multiple targets)"
            )

        # A target that repeats, or equals the source, would make the
        # competing-share compensation describe an unintended model.
        if self._str_source_allele in targets:
            raise ValueError(
                f"PointMutation {self.name!r}: target allele "
                f"{self._str_source_allele!r} equals the source allele"
            )
        if len(set(targets)) != len(targets):
            raise ValueError(
                f"PointMutation {self.name!r}: target_alleles must be distinct, "
                f"got {list(targets)}"
            )
        self._str_target_alleles: Tuple[str, ...] = targets

        if mutation_rate is not None and mutation_rates is not None:
            raise ValueError(
                f"PointMutation {self.name!r}: declare either mutation_rate or "
                "mutation_rates, not both"
            )
        declarations: Tuple[SexSpecificRates, ...]
        if mutation_rate is not None:
            declarations = (mutation_rate,)
        elif mutation_rates is not None:
            declarations = tuple(mutation_rates)
        else:
            raise ValueError(
                f"PointMutation {self.name!r}: declare mutation_rate (single "
                "target) or mutation_rates (one per target)"
            )
        if len(declarations) != len(targets):
            raise ValueError(
                f"PointMutation {self.name!r}: expected {len(targets)} rate "
                f"declaration(s), one per target, got {len(declarations)}"
            )
        # Scalar/dict/pair declarations normalize to one (female, male) pair per
        # target; the canonical attribute is a sequence of such pairs.
        self.mutation_rates: Tuple[Tuple[float, float], ...] = tuple(
            _resolve_rate_pair(declaration, self.name)
            for declaration in declarations
        )

        self.rate_mode: RateMode = rate_mode

        # Store declarative fitness scaling configs; they apply to every target.
        self.viability_scaling = viability_scaling
        self.fecundity_scaling = fecundity_scaling
        self.sexual_selection_scaling = sexual_selection_scaling
        self.zygote_viability_scaling = zygote_viability_scaling
        self.viability_mode: AlleleScalingMode = viability_mode
        self.fecundity_mode: AlleleScalingMode = fecundity_mode
        self.sexual_selection_mode: AlleleScalingMode = sexual_selection_mode
        self.zygote_viability_mode: AlleleScalingMode = zygote_viability_mode

        # Fail at configuration time: a rate declaration that cannot compete
        # (sum above 1 in "strict" mode) or an invalid rate_mode must not sit on
        # a preset until build.  reconfigure_preset re-runs the same checks.
        self.effective_rates()

    def _declared_rate_pairs(self) -> List[Tuple[float, float]]:
        """Return ``mutation_rates`` normalized to one pair per target.

        ``reconfigure_preset`` writes straight to the attribute, so the value
        may be any shape accepted at construction: a bare number or mapping is
        read as the single-target declaration.

        Raises:
            ValueError: If the value is not a sequence with one declaration
                per target, or a declaration is malformed.
        """
        declarations = _rate_declarations(self.mutation_rates, self.name)
        expected = len(self._str_target_alleles)
        if len(declarations) != expected:
            raise ValueError(
                f"PointMutation {self.name!r}: expected {expected} rate "
                f"declaration(s), got {len(declarations)}"
            )
        return [
            _resolve_rate_pair(declaration, self.name)
            for declaration in declarations
        ]

    def effective_rates(self) -> Tuple[Tuple[float, float], ...]:
        """Return the germline rates actually handed to the conversion cascade.

        Returns:
            One ``(female_rate, male_rate)`` pair per target in declaration
            order.  The rates are already compensated for the cascade, so each
            target's effective germline share equals its declared rate.

        Raises:
            ValueError: If a declared rate is invalid, or the declared rates
                cannot compete (their sum exceeds 1 in ``"strict"`` mode).
        """
        declared = self._declared_rate_pairs()
        context = f"PointMutation {self.name!r}"
        # The cascade runs per (sex, parent) row, so the compensation is
        # computed independently for each sex.
        by_sex = [
            _correct_competing(
                [pair[sex] for pair in declared],
                self.rate_mode,
                context,
            )
            for sex in (Sex.FEMALE, Sex.MALE)
        ]
        female_rates, male_rates = by_sex
        return tuple((female, male) for female, male in zip(female_rates, male_rates))

    @property
    def source_allele(self) -> Gene:
        """Gene: The allele that mutates."""
        return self._resolve_bound_gene(self._str_source_allele)

    @property
    def target_alleles(self) -> Tuple[Gene, ...]:
        """Tuple[Gene, ...]: The target alleles, in declaration order."""
        return tuple(
            self._resolve_bound_gene(name) for name in self._str_target_alleles
        )

    def fitness_patch(self) -> PresetFitnessPatch:
        """Return the declarative fitness patch for the target alleles."""
        # One patch entry covers every target; the fitness applier counts
        # combined copies, so scaling composes across targets.
        return make_fitness_patch_given_allele_scaling(
            list(self._str_target_alleles),
            self.viability_scaling,
            self.fecundity_scaling,
            self.sexual_selection_scaling,
            self.zygote_viability_scaling,
            self.viability_mode,
            self.fecundity_mode,
            self.sexual_selection_mode,
            self.zygote_viability_mode,
        )

    def gamete_modifier(self, host: RecipeHost) -> Optional[GameteModifier]:
        """Implement germline point mutation with competing targets.

        Every gamete carrying the source allele converts, regardless of the
        producer's genotype: the rules carry no parent-genotype filter.
        """
        rule_set = GameteConversionRuleSet(f"{self.name}_GermlineMutation")

        # Rules are appended target by target so the cascade order matches the
        # declared order the compensation was computed for.
        for target, rates in zip(self._str_target_alleles, self.effective_rates()):
            for sex in (Sex.FEMALE, Sex.MALE):
                rate = rates[sex]
                # A zero rate adds no rule, keeping the cascade minimal.
                if rate > 0:
                    rule_set.add_allele_convert(
                        from_allele=self._str_source_allele,
                        to_allele=target,
                        rate=rate,
                        filters={
                            "parent_sex": "female" if sex == Sex.FEMALE else "male"
                        },
                    )

        if not rule_set.rules:
            return None
        return rule_set.to_gamete_modifier(host)  # type: ignore[return-type]  # structurally satisfies the GameteModifier protocol

    def zygote_modifier(self, host: RecipeHost) -> Optional[ZygoteModifier]:
        """Return None: this preset mutates germline gametes only.

        The embryonic (zygote-stage) channel is deliberately deferred, so the
        preset registers no zygote modifier at all; ``host`` is accepted to
        satisfy the ``GeneticPreset`` contract.
        """
        return None
