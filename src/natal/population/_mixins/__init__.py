"""Mixins for BasePopulation — decompose the population ABC into focused concerns.

This package extracts reusable behaviour from
:mod:`natal.population.base` into separate mixin modules:

- ``_hooks``: Hook registration, compilation, and lifecycle integration.
- ``_modifiers``: Modifier and preset management.
- ``_observation``: Observation mask building and history recording.
- ``_output``: Query, export, and lifecycle-orchestration methods.
"""
