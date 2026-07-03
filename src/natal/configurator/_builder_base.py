"""Base builder class extracted from _factory.py."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

import natal.data as _population_config
from natal.genetics import Genotype, Species
from natal.output.observation import GroupsInput
from natal.utils.helpers import resolve_sex_label
from natal.utils.types import Sex

if TYPE_CHECKING:
    pass


GenotypeSelectorAtom = Union[Genotype, str]
GenotypeSelector = Union[GenotypeSelectorAtom, Tuple[GenotypeSelectorAtom, ...]]
ArrayF64 = NDArray[np.float64]
InitialAgeCountValue: TypeAlias = (
    Sequence[float | int] | Mapping[int, float | int] | ArrayF64 | int | float
)
InitialIndividualCountInput: TypeAlias = Mapping[str, Mapping[Any, InitialAgeCountValue]]
InitialSpermStorageInput: TypeAlias = Mapping[Any, Mapping[Any, InitialAgeCountValue]]
HookFn = Callable[..., object]
ModifierSpec = Tuple[int, Optional[str], HookFn]
HookRegistration = Tuple[HookFn, Optional[str], Optional[int]]
HookMap = Dict[str, List[HookRegistration]]
SexScalarMap = Dict[str, float]
AgeScalarMap = Dict[int, float]
ViabilityNestedMap = Dict[Union[str, Sex, int], Union[float, AgeScalarMap]]
ViabilityMap = Dict[GenotypeSelector, Union[float, ViabilityNestedMap]]
FecundityMap = Dict[GenotypeSelector, Union[float, SexScalarMap]]
SexualSelectionMap = Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
ZygoteViabilityMap = Dict[GenotypeSelector, Union[float, SexScalarMap]]
FitnessOperationName = Literal["viability", "fecundity", "sexual_selection", "zygote_viability"]
FitnessOperation = Tuple[FitnessOperationName, Tuple[object], Dict[str, str]]

InitializeMapFn = Callable[..., NDArray[np.float64]]
initialize_gamete_map = cast(InitializeMapFn, _population_config.initialize_gamete_map)
initialize_zygote_map = cast(InitializeMapFn, _population_config.initialize_zygote_map)


class PopulationBuilderBase:
    """Abstract base builder with common chainable methods.

    Attributes:
        species (Species): Genetic architecture for the population.
    """

    def __init__(self, species: Species):
        """Initialize builder with required species.

        Args:
            species (Species): Genetic architecture for the population.
        """
        self.species = species
        self._presets: List[Any] = []
        self._observation_groups: Optional[GroupsInput] = None
        self._observation_collapse_age: bool = False
        self._params: dict[str, object] = {}
        self._custom_specs: dict[str, bool | int | float | NDArray[np.float64]] = {}

    def _param(self, key: str, default: object = None) -> object:
        return self._params.get(key, default)

    def _set_param(self, key: str, value: object) -> None:
        self._params[key] = value

    def _set_params(self, domain: str, **kwargs: object) -> None:
        for k, v in kwargs.items():
            self._params[f"{domain}.{k}"] = v

    def get_params(self) -> dict[str, object]:
        return dict(self._params)

    def with_observation(
        self,
        groups: GroupsInput,
        *,
        collapse_age: bool = False,
    ) -> PopulationBuilderBase:
        """Register observation groups for compressed history recording.

        The groups are compiled into a binary mask and passed to the
        simulation kernel on each ``run()`` call. Once set, history records
        store aggregated observation data instead of raw flattened state.

        Args:
            groups: Observation groups (dict of name -> spec, list of specs,
                or None for one-group-per-genotype).
            collapse_age: Whether to collapse the age axis in exports.

        Returns:
            PopulationBuilderBase: Self for chaining.
        """
        self._observation_groups = groups
        self._observation_collapse_age = collapse_age
        return self

    def custom(
        self,
        **kwargs: bool | int | float | NDArray[np.float64],
    ) -> PopulationBuilderBase:
        """Register custom named fields stored on ``config.custom``.

        Scalar values (float, int) become scalar fields.
        ndarray values become a (sex, age, genotype) shaped sub-array.

        Usage::

            builder.custom(temperature=25.0, habitat=habitat_array)

        Args:
            **kwargs: Field name -> value. Scalars or 3-D ndarray.

        Returns:
            PopulationBuilderBase: Self for chaining.
        """
        for name, value in kwargs.items():
            self._custom_specs[name] = value
        return self

    @staticmethod
    def _resolve_viability_age(age_key: object, n_ages: int) -> int:
        """Resolve and validate a viability age key.

        Args:
            age_key (object): Candidate age key.
            n_ages (int): Number of available age classes.

        Returns:
            int: Validated age index.

        Raises:
            TypeError: If age key is not an integer.
            ValueError: If age key is out of range.
        """
        if not isinstance(age_key, int) or isinstance(age_key, bool):
            raise TypeError(f"viability age key must be int, got {type(age_key)}")
        if age_key < 0 or age_key >= n_ages:
            raise ValueError(f"viability age key {age_key} out of range [0, {n_ages})")
        return age_key

    @staticmethod
    def _iter_viability_updates(
        values: Union[float, ViabilityNestedMap],
        n_ages: int,
        default_age: int,
    ) -> List[Tuple[int, int, float]]:
        """Expand viability value specs into (sex_idx, age_idx, value) triples.

        Supported forms:
            - float: applies to both sexes at default_age.
            - {"female": 0.9, "male": 0.8}: per-sex at default_age.
            - {0: 0.95, 1: 0.85}: both sexes, age-specific.
            - {"female": {0: 0.95}, "male": {1: 0.9}}: sex+age specific.

        Args:
            values (Union[float, ViabilityNestedMap]): Viability value specification.
            n_ages (int): Number of age classes.
            default_age (int): Fallback age when age is not provided.

        Returns:
            List[Tuple[int, int, float]]: Expanded updates.

        Raises:
            TypeError: If input structure is unsupported.
            ValueError: If map is empty or age keys are invalid.
        """
        if not isinstance(values, dict):
            scalar = float(values)
            return [(0, default_age, scalar), (1, default_age, scalar)]

        if not values:
            raise ValueError("viability mapping cannot be empty")

        updates: List[Tuple[int, int, float]] = []
        for key, key_value in values.items():
            if isinstance(key, int) and not isinstance(key, bool):
                if isinstance(key_value, dict):
                    raise TypeError("age-based viability values must be numeric")
                age_idx = PopulationBuilderBase._resolve_viability_age(key, n_ages)
                val = float(key_value)
                updates.append((0, age_idx, val))
                updates.append((1, age_idx, val))
                continue

            if isinstance(key, (str, Sex)):
                sex_idx = int(key.value) if isinstance(key, Sex) else resolve_sex_label(key)
                if isinstance(key_value, dict):
                    for age_key, age_value in key_value.items():
                        age_idx = PopulationBuilderBase._resolve_viability_age(age_key, n_ages)
                        updates.append((sex_idx, age_idx, float(age_value)))
                else:
                    updates.append((sex_idx, default_age, float(key_value)))
                continue

            raise TypeError(
                "viability map keys must be sex labels (str/Sex) or age indices (int)"
            )

        return updates

    def add_preset(self, preset: Any) -> PopulationBuilderBase:
        """Add a gene drive preset to apply during build.

        Presets are applied in the order they are added.

        Args:
            preset (Any): A GeneDrivePreset or similar modification system.

        Returns:
            PopulationBuilderBase: Self for chaining.
        """
        self._presets.append(preset)
        return self

    def build(self) -> Any:
        """Build and return the configured Population.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
