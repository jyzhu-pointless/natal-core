"""Simulation algorithm helpers for engine lifecycle stages.

This package provides Numba-accelerated algorithm functions separated
by lifecycle model: age-structured (cohort-based) and discrete-generation.

Modules:
    age_structured: Absolute-population-size algorithms for mating, sperm
        storage, offspring distribution, survival, and aging.
    discrete_generation: Non-overlapping generation algorithms for mating
        allocation and fertilization.
"""
