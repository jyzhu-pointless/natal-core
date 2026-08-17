//! Python-side simulation configuration copied into a plain Rust struct.
//!
//! ``EngineSession::new`` reads the ``PopulationConfig`` attributes once and
//! owns plain ``Vec`` copies.  The per-tick kernels never touch Python objects.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone)]
pub struct SimConfig {
    pub n_ages: usize,
    pub n_ztypes: usize,
    pub adult_start_age: usize,
    pub new_adult_age: usize,
    pub stochastic: bool,
    pub continuous_sampling: bool,
    pub fixed_egg_count: bool,
    pub has_sex_chromosomes: bool,
    pub eggs_per_female: f64,
    pub sperm_displacement_rate: f64,
    pub sex_ratio: f64,
    pub carrying_capacity: f64,
    pub expected_competition_strength: f64,
    pub expected_survival_rate: f64,
    pub low_density_growth_rate: f64,
    pub juvenile_growth_mode: i64,
    pub age_based_mating_rates: Vec<f64>,
    pub age_based_reproduction_rates: Vec<f64>,
    pub female_age_based_fertility: Vec<f64>,
    pub age_based_survival_rates: Vec<f64>,
    pub viability_fitness: Vec<f64>,
    pub fecundity_fitness: Vec<f64>,
    pub sexual_selection_fitness: Vec<f64>,
    pub zygote_viability_fitness: Vec<f64>,
    pub age_based_relative_competition_strength: Vec<f64>,
    pub adult_ages: Vec<usize>,
    pub offspring_tensor: Vec<f64>,
    pub female_ztype_compatibility: Vec<f64>,
    pub male_ztype_compatibility: Vec<f64>,
    pub female_only_by_sex_chrom: Vec<bool>,
    pub male_only_by_sex_chrom: Vec<bool>,
}

fn shape_error(name: &str, expected: &str, got: &[usize]) -> PyErr {
    PyValueError::new_err(format!("{name} must have shape {expected}, got {got:?}"))
}

fn extract_f64_1d(obj: &Bound<'_, PyAny>, name: &str, expected: usize) -> PyResult<Vec<f64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, f64>>()?;
    if array.len() != expected {
        return Err(shape_error(name, &format!("({expected},)"), array.shape()));
    }
    Ok(array.as_slice()?.to_vec())
}

fn extract_f64_2d(
    obj: &Bound<'_, PyAny>,
    name: &str,
    rows: usize,
    cols: usize,
) -> PyResult<Vec<f64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray2<'_, f64>>()?;
    let shape = array.shape();
    if shape.len() != 2 || shape[0] != rows || shape[1] != cols {
        return Err(shape_error(name, &format!("({rows}, {cols})"), shape));
    }
    Ok(array.as_slice()?.to_vec())
}

fn extract_f64_3d(
    obj: &Bound<'_, PyAny>,
    name: &str,
    d0: usize,
    d1: usize,
    d2: usize,
) -> PyResult<Vec<f64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray3<'_, f64>>()?;
    let shape = array.shape();
    if shape.len() != 3 || shape[0] != d0 || shape[1] != d1 || shape[2] != d2 {
        return Err(shape_error(name, &format!("({d0}, {d1}, {d2})"), shape));
    }
    Ok(array.as_slice()?.to_vec())
}

fn extract_bool_1d(obj: &Bound<'_, PyAny>, name: &str, expected: usize) -> PyResult<Vec<bool>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, bool>>()?;
    if array.len() != expected {
        return Err(shape_error(name, &format!("({expected},)"), array.shape()));
    }
    Ok(array.as_slice()?.to_vec())
}

fn extract_i64_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, i64>>()?;
    Ok(array.as_slice()?.to_vec())
}

fn extract_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(scalar);
    }
    let scalar = value.call_method0("item")?;
    scalar.extract::<f64>()
}

fn extract_bool(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    obj.getattr(name)?.extract::<bool>()
}

fn extract_i64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<i64>() {
        return Ok(scalar);
    }
    let scalar = value.call_method0("item")?;
    scalar.extract::<i64>()
}

impl SimConfig {
    pub fn from_python(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n_ages = extract_i64(config, "n_ages")? as usize;
        let n_ztypes = extract_i64(config, "n_ztypes")? as usize;
        let new_adult_age = extract_i64(config, "new_adult_age")? as usize;
        if n_ages == 0 || n_ztypes == 0 {
            return Err(PyValueError::new_err(
                "n_ages and n_ztypes must be positive",
            ));
        }
        if new_adult_age == 0 || new_adult_age > n_ages {
            return Err(PyValueError::new_err(format!(
                "new_adult_age must be in [1, {n_ages}], got {new_adult_age}"
            )));
        }

        let adult_ages_raw = extract_i64_1d(config, "adult_ages")?;
        let adult_ages: Vec<usize> = adult_ages_raw.iter().map(|&v| v as usize).collect();
        let adult_start_age = *adult_ages.first().unwrap_or(&0);

        let mut cfg = Self {
            n_ages,
            n_ztypes,
            adult_start_age,
            new_adult_age,
            stochastic: extract_bool(config, "stochastic")?,
            continuous_sampling: extract_bool(config, "continuous_sampling")?,
            fixed_egg_count: extract_bool(config, "fixed_egg_count")?,
            has_sex_chromosomes: extract_bool(config, "has_sex_chromosomes")?,
            eggs_per_female: extract_f64(config, "eggs_per_female")?,
            sperm_displacement_rate: extract_f64(config, "sperm_displacement_rate")?,
            sex_ratio: extract_f64(config, "sex_ratio")?,
            carrying_capacity: extract_f64(config, "carrying_capacity")?,
            expected_competition_strength: extract_f64(config, "expected_competition_strength")?,
            expected_survival_rate: extract_f64(config, "expected_survival_rate")?,
            low_density_growth_rate: extract_f64(config, "low_density_growth_rate")?,
            juvenile_growth_mode: extract_i64(config, "juvenile_growth_mode")?,
            age_based_mating_rates: extract_f64_2d(config, "age_based_mating_rates", 2, n_ages)?,
            age_based_reproduction_rates: extract_f64_1d(
                config,
                "age_based_reproduction_rates",
                n_ages,
            )?,
            female_age_based_fertility: extract_f64_1d(
                config,
                "female_age_based_fertility",
                n_ages,
            )?,
            age_based_survival_rates: extract_f64_2d(
                config,
                "age_based_survival_rates",
                2,
                n_ages,
            )?,
            viability_fitness: extract_f64_3d(config, "viability_fitness", 2, n_ages, n_ztypes)?,
            fecundity_fitness: extract_f64_2d(config, "fecundity_fitness", 2, n_ztypes)?,
            sexual_selection_fitness: extract_f64_2d(
                config,
                "sexual_selection_fitness",
                n_ztypes,
                n_ztypes,
            )?,
            zygote_viability_fitness: extract_f64_2d(
                config,
                "zygote_viability_fitness",
                2,
                n_ztypes,
            )?,
            age_based_relative_competition_strength: extract_f64_1d(
                config,
                "age_based_relative_competition_strength",
                n_ages,
            )?,
            adult_ages,
            offspring_tensor: extract_f64_3d(
                config,
                "offspring_tensor",
                n_ztypes,
                n_ztypes,
                n_ztypes,
            )?,
            female_ztype_compatibility: extract_f64_1d(
                config,
                "female_ztype_compatibility",
                n_ztypes,
            )?,
            male_ztype_compatibility: extract_f64_1d(config, "male_ztype_compatibility", n_ztypes)?,
            female_only_by_sex_chrom: extract_bool_1d(
                config,
                "female_only_by_sex_chrom",
                n_ztypes,
            )?,
            male_only_by_sex_chrom: extract_bool_1d(config, "male_only_by_sex_chrom", n_ztypes)?,
        };
        cfg.validate();
        Ok(cfg)
    }

    fn validate(&mut self) {
        self.eggs_per_female = self.eggs_per_female.max(0.0);
        self.sperm_displacement_rate = crate::rng::clamp01(self.sperm_displacement_rate);
        self.sex_ratio = crate::rng::clamp01(self.sex_ratio);
    }
}
