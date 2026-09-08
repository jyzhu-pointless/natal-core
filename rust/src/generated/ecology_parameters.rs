//! GENERATED FILE — DO NOT EDIT.
//!
//! Produced by ``scripts/generate_param_tables.py`` from
//! ``src/natal/parameters.jsonc`` (single source of truth, plan
//! section 5.4).  Wire order follows ``ECO_PARAM_NAMES`` in
//! ``natal/frontend/hooks/types.py``.  Run the generator after editing
//! the jsonc; the pytest suite fails on drift.

/// Wire names of the runtime-mutable ecology parameters, in the
/// fixed order shared with the Python hook compiler.
pub const ECO_PARAM_COLUMNS: [&str; 5] = [
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
];

/// Number of wire columns.
pub const N_ECO_PARAMS: usize = ECO_PARAM_COLUMNS.len();

/// Validity bounds per column — the generated mirror of the jsonc
/// scalar ``bounds`` (same order as ``ECO_PARAM_COLUMNS``).
pub const ECO_PARAM_BOUNDS: [(f64, f64); N_ECO_PARAMS] = [
    (0.0, 1000000000000.0),
    (0.0, 1000000.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1000000.0),
];

/// Ecology scalar channel names of the params contract: the wire
/// order plus ``external_expected_eggs`` appended last.
pub const ECOLOGY_SCALAR_COLUMNS: [&str; 6] = [
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
    "external_expected_eggs",
];

/// Ecology scalar names carried by a memory checkpoint: the wire
/// order plus ``growth_mode`` and ``external_expected_eggs``.
pub const ECOLOGY_SCALARS: [&str; 7] = [
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
    "growth_mode",
    "external_expected_eggs",
];
