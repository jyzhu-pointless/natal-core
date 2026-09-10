//! The flat per-tick kernel configuration, assembled from owned contracts.
//!
//! Kernels consume [`AgeStructuredConfig`] and never see Python objects.  Sessions no
//! longer snapshot a Python config: they own a
//! [`contract::Blueprint`](crate::model::blueprint::Blueprint) plus
//! [`contract::EcologyParams`](crate::model::ecology::EcologyParams) (columnized ecology) and a
//! [`contract::GeneticsTensors`](crate::model::genetics::GeneticsTensors) (genetics), and
//! assemble a fresh ``AgeStructuredConfig`` view at every tick-batch entry point, so
//! parameter writes take effect on the next batch without any session
//! rebuild.  Spatial sessions assemble one view *per deme* from the deme's
//! ecology column entry and its shared genetics variant.

use crate::kernels::equilibrium::equilibrium_metrics;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Flat kernel view assembled from the owned contracts.
///
/// This is a pure-copy structure (tens of KB): every field either mirrors a
/// contract value or is derived on demand from it.  The derived equilibrium
/// metrics ``expected_competition_strength`` / ``expected_survival_rate``
/// are recomputed from the current params at every [`AgeStructuredConfig::assemble`]
/// call, replacing the stored draft fields of the legacy ``from_python``
/// path.
///
/// ## Notes
/// - All array fields are stored in row-major flat ``Vec`` layout.
/// - Scalar fields are normalized by [`AgeStructuredConfig::assemble`] before first use.
#[derive(Clone)]
pub struct AgeStructuredConfig {
    // --- Dimensions ---
    pub n_ages: usize,
    pub n_ztypes: usize,
    pub adult_start_age: usize,
    pub new_adult_age: usize,

    // --- Sampling flags ---
    pub stochastic: bool,
    pub continuous_sampling: bool,
    pub fixed_egg_count: bool,
    pub has_sex_chromosomes: bool,

    // --- Scalar demographic rates ---
    pub eggs_per_female: f64,
    pub sperm_displacement_rate: f64,
    pub sex_ratio: f64,
    pub carrying_capacity: f64,
    pub expected_competition_strength: f64,
    pub expected_survival_rate: f64,
    pub low_density_growth_rate: f64,
    pub juvenile_growth_mode: i64,

    // --- Age/sex structured arrays ---
    pub age_based_mating_rates: Vec<f64>,
    pub age_based_reproduction_rates: Vec<f64>,
    pub female_age_based_fertility: Vec<f64>,
    pub age_based_survival_rates: Vec<f64>,

    // --- Fitness arrays ---
    pub viability_fitness: Vec<f64>,
    pub fecundity_fitness: Vec<f64>,
    pub sexual_selection_fitness: Vec<f64>,
    pub zygote_viability_fitness: Vec<f64>,
    pub age_based_relative_competition_strength: Vec<f64>,
    pub adult_ages: Vec<usize>,

    // --- Inheritance / sex-chromosome arrays ---
    pub offspring_tensor: Vec<f64>,
    pub female_ztype_compatibility: Vec<f64>,
    pub male_ztype_compatibility: Vec<f64>,
    pub female_only_by_sex_chrom: Vec<bool>,
    pub male_only_by_sex_chrom: Vec<bool>,
}

/// Copy one deme's segment of a columnized vector into a flat kernel array.
fn deme_segment(column: &[f64], deme: usize, per_deme: usize) -> Vec<f64> {
    let start = deme * per_deme;
    column[start..start + per_deme].to_vec()
}

impl AgeStructuredConfig {
    /// Assemble a kernel config from the owned contracts at deme 0.
    ///
    /// Panmictic and homogeneous callers own length-1 (or tiled) columns,
    /// so deme 0 carries the values every deme consumes.
    ///
    /// ## Errors
    /// Same as [`AgeStructuredConfig::assemble_deme`].
    pub fn assemble(
        bp: &Blueprint,
        params: &EcologyParams,
        genetics: &GeneticsTensors,
    ) -> PyResult<Self> {
        Self::assemble_deme(bp, params, genetics, 0)
    }

    /// Assemble a kernel config from the owned contracts for one deme.
    ///
    /// Validates the blueprint dimensions, derives the equilibrium metrics
    /// from the *current* params (bit-for-bit port of the Python
    /// ``compute_equilibrium_metrics``), and copies the deme's ecology
    /// column segment plus the shared genetics tables.
    ///
    /// ## Parameters
    /// - `bp`: The frozen blueprint.
    /// - `params`: The current runtime parameters (columnized ecology).
    /// - `genetics`: The genetics tables (variant shared across demes).
    /// - `deme`: Deme whose ecology column feeds the view.
    ///
    /// ## Returns
    /// A ``AgeStructuredConfig`` view consistent with the current contract values.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when dimensions are invalid, the deme index
    /// is out of range, or a params vector disagrees with the
    /// blueprint-declared size.
    pub fn assemble_deme(
        bp: &Blueprint,
        params: &EcologyParams,
        genetics: &GeneticsTensors,
        deme: usize,
    ) -> PyResult<Self> {
        let n_ages = bp.n_ages;
        let n_ztypes = bp.n_ztypes;
        if n_ages == 0 || n_ztypes == 0 {
            return Err(PyValueError::new_err(
                "n_ages and n_ztypes must be positive",
            ));
        }
        if bp.new_adult_age == 0 || bp.new_adult_age > n_ages {
            return Err(PyValueError::new_err(format!(
                "new_adult_age must be in [1, {n_ages}], got {}",
                bp.new_adult_age
            )));
        }
        if deme >= params.n_demes {
            return Err(PyValueError::new_err(format!(
                "deme {deme} out of range for {} ecology columns",
                params.n_demes
            )));
        }
        params.validate(bp)?;
        genetics.validate(bp)?;
        let a = n_ages;

        // Derive the equilibrium metrics from current params so parameter
        // changes take effect without any stored, stale copy.
        let (expected_competition_strength, expected_survival_rate) =
            equilibrium_metrics(bp, params, deme);

        let deme0 = |column: &Vec<f64>| -> f64 { column[deme] };
        let cfg = Self {
            n_ages,
            n_ztypes,
            adult_start_age: *bp.adult_ages.first().unwrap_or(&0) as usize,
            new_adult_age: bp.new_adult_age,
            stochastic: bp.stochastic,
            continuous_sampling: bp.continuous_sampling,
            fixed_egg_count: bp.fixed_egg_count,
            has_sex_chromosomes: bp.has_sex_chromosomes,
            eggs_per_female: deme0(&params.eggs_per_female).max(0.0),
            sperm_displacement_rate: crate::kernels::rng::clamp01(deme0(
                &params.sperm_displacement_rate,
            )),
            sex_ratio: crate::kernels::rng::clamp01(deme0(&params.sex_ratio)),
            carrying_capacity: deme0(&params.carrying_capacity),
            expected_competition_strength,
            expected_survival_rate,
            low_density_growth_rate: deme0(&params.low_density_growth_rate),
            juvenile_growth_mode: params.growth_mode[deme],
            age_based_mating_rates: deme_segment(&params.mating_rates, deme, 2 * a),
            age_based_reproduction_rates: deme_segment(&params.reproduction_rates, deme, a),
            female_age_based_fertility: deme_segment(&params.fertility, deme, a),
            age_based_survival_rates: deme_segment(&params.survival_rates, deme, 2 * a),
            viability_fitness: genetics.viability_fitness.clone(),
            fecundity_fitness: genetics.fecundity_fitness.clone(),
            sexual_selection_fitness: genetics.sexual_selection_fitness.clone(),
            zygote_viability_fitness: genetics.zygote_viability_fitness.clone(),
            age_based_relative_competition_strength: deme_segment(
                &params.competition_weights,
                deme,
                a,
            ),
            adult_ages: bp.adult_ages.iter().map(|&v| v as usize).collect(),
            offspring_tensor: genetics.offspring_tensor.clone(),
            female_ztype_compatibility: genetics.female_ztype_compatibility.clone(),
            male_ztype_compatibility: genetics.male_ztype_compatibility.clone(),
            female_only_by_sex_chrom: bp.female_only_by_sex_chrom.clone(),
            male_only_by_sex_chrom: bp.male_only_by_sex_chrom.clone(),
        };
        Ok(cfg)
    }
}
