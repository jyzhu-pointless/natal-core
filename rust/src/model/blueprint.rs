//! The frozen model specification.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::model::python::{
    extract_bool, extract_bool_vec, extract_f64_vec, extract_i64, extract_i64_vec,
    extract_string_vec,
};

/// The frozen model specification.
#[derive(Clone)]
#[pyclass]
pub struct Blueprint {
    // -- dimensions --
    pub n_sexes: usize,
    pub n_ages: usize,
    pub n_ztypes: usize,
    pub n_gtypes: usize,
    pub n_glabs: usize,
    pub new_adult_age: usize,
    pub adult_ages: Vec<i64>,
    // -- execution flags --
    pub stochastic: bool,
    pub continuous_sampling: bool,
    pub fixed_egg_count: bool,
    pub has_sex_chromosomes: bool,
    pub extreme_speed_mode: i64,
    // -- symbolic name directory --
    pub ztype_names: Vec<String>,
    pub gtype_names: Vec<String>,
    // -- species-derived masks --
    pub female_only_by_sex_chrom: Vec<bool>,
    pub male_only_by_sex_chrom: Vec<bool>,
    // -- initial population --
    pub initial_individual_count: Vec<f64>,
    pub initial_sperm_storage: Vec<f64>,
    // -- spatial domain --
    pub n_demes: usize,
    pub migration_indptr: Vec<i64>,
    pub migration_dest_idx: Vec<i64>,
    pub migration_weights: Vec<f64>,
}

#[pymethods]
impl Blueprint {
    #[new]
    #[allow(clippy::too_many_arguments)] // from_parts handoff mirrors the Python contract 1:1
    #[pyo3(signature = (n_sexes, n_ages, n_ztypes, n_gtypes, n_glabs, new_adult_age,
                        adult_ages, stochastic, continuous_sampling, fixed_egg_count,
                        has_sex_chromosomes, extreme_speed_mode, ztype_names, gtype_names,
                        female_only_by_sex_chrom, male_only_by_sex_chrom,
                        initial_individual_count, initial_sperm_storage,
                        n_demes, migration_indptr, migration_dest_idx, migration_weights))]
    #[allow(clippy::fn_params_excessive_bools)] // the contract itself has five flags
    fn new(
        n_sexes: usize,
        n_ages: usize,
        n_ztypes: usize,
        n_gtypes: usize,
        n_glabs: usize,
        new_adult_age: usize,
        adult_ages: Vec<i64>,
        stochastic: bool,
        continuous_sampling: bool,
        fixed_egg_count: bool,
        has_sex_chromosomes: bool,
        extreme_speed_mode: i64,
        ztype_names: Vec<String>,
        gtype_names: Vec<String>,
        female_only_by_sex_chrom: Vec<bool>,
        male_only_by_sex_chrom: Vec<bool>,
        initial_individual_count: Vec<f64>,
        initial_sperm_storage: Vec<f64>,
        n_demes: usize,
        migration_indptr: Vec<i64>,
        migration_dest_idx: Vec<i64>,
        migration_weights: Vec<f64>,
    ) -> Self {
        Self {
            n_sexes,
            n_ages,
            n_ztypes,
            n_gtypes,
            n_glabs,
            new_adult_age,
            adult_ages,
            stochastic,
            continuous_sampling,
            fixed_egg_count,
            has_sex_chromosomes,
            extreme_speed_mode,
            ztype_names,
            gtype_names,
            female_only_by_sex_chrom,
            male_only_by_sex_chrom,
            initial_individual_count,
            initial_sperm_storage,
            n_demes,
            migration_indptr,
            migration_dest_idx,
            migration_weights,
        }
    }

    /// Validate internal shape consistency against the declared dimensions.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when any array length disagrees with the
    /// declared dimensions.
    pub fn validate(&self) -> PyResult<()> {
        let z = self.n_ztypes;
        let a = self.n_ages;
        let checks: [(&str, usize, usize); 6] = [
            ("ztype_names", self.ztype_names.len(), z),
            ("gtype_names", self.gtype_names.len(), self.n_gtypes),
            (
                "female_only_by_sex_chrom",
                self.female_only_by_sex_chrom.len(),
                z,
            ),
            (
                "male_only_by_sex_chrom",
                self.male_only_by_sex_chrom.len(),
                z,
            ),
            (
                "initial_individual_count",
                self.initial_individual_count.len(),
                2 * a * z,
            ),
            (
                "initial_sperm_storage",
                self.initial_sperm_storage.len(),
                if self.initial_sperm_storage.is_empty() {
                    0
                } else {
                    a * z * z
                },
            ),
        ];
        // adult_ages must be exactly the adult range of the age axis.
        let expected_adults: Vec<i64> = (self.new_adult_age as i64..self.n_ages as i64).collect();
        if self.adult_ages != expected_adults {
            return Err(PyValueError::new_err(format!(
                "Blueprint.adult_ages: expected {expected_adults:?}, got {:?}",
                self.adult_ages
            )));
        }
        for (name, got, expected) in checks {
            if got != expected {
                return Err(PyValueError::new_err(format!(
                    "Blueprint.{name}: expected {expected} elements, got {got}"
                )));
            }
        }
        // Migration CSR: row pointer geometry and destination range.  The
        // panmictic default (one deme, all CSR arrays empty) is the legal
        // "no spatial routing" sentinel.
        let is_panmictic_sentinel = self.n_demes == 1
            && self.migration_indptr.is_empty()
            && self.migration_dest_idx.is_empty();
        if is_panmictic_sentinel {
            return Ok(());
        }
        if self.migration_indptr.len() != self.n_demes + 1 {
            return Err(PyValueError::new_err(format!(
                "Blueprint.migration_indptr: expected {} elements, got {}",
                self.n_demes + 1,
                self.migration_indptr.len()
            )));
        }
        if self.migration_indptr.first() != Some(&0) {
            return Err(PyValueError::new_err(
                "Blueprint.migration_indptr must start at 0",
            ));
        }
        if let Some(pair) = self.migration_indptr.windows(2).find(|w| w[1] < w[0]) {
            return Err(PyValueError::new_err(format!(
                "Blueprint.migration_indptr must be non-decreasing ({} > {})",
                pair[0], pair[1]
            )));
        }
        let nnz = *self.migration_indptr.last().unwrap_or(&0);
        if self.migration_dest_idx.len() != nnz as usize
            || self.migration_weights.len() != nnz as usize
        {
            return Err(PyValueError::new_err(format!(
                "Blueprint.migration CSR: indptr ends at {nnz} but dest_idx has {} and weights {} entries",
                self.migration_dest_idx.len(),
                self.migration_weights.len()
            )));
        }
        for &dst in &self.migration_dest_idx {
            if dst < 0 || dst as usize >= self.n_demes {
                return Err(PyValueError::new_err(format!(
                    "Blueprint.migration_dest_idx: destination {dst} out of range for {n} demes",
                    n = self.n_demes
                )));
            }
        }
        Ok(())
    }

    // -- frozen readouts (plain values; arrays come back as copies) --

    /// Number of age classes.
    #[getter]
    fn n_ages(&self) -> usize {
        self.n_ages
    }

    /// Number of zygote types.
    #[getter]
    fn n_ztypes(&self) -> usize {
        self.n_ztypes
    }

    /// First adult age class.
    #[getter]
    fn new_adult_age(&self) -> usize {
        self.new_adult_age
    }

    /// Whether demographic events are stochastic.
    #[getter]
    fn stochastic(&self) -> bool {
        self.stochastic
    }

    /// Wright-Fisher fused-tick selector.
    #[getter]
    fn extreme_speed_mode(&self) -> i64 {
        self.extreme_speed_mode
    }

    /// The zygote-type name directory (copy).
    #[getter]
    fn ztype_names(&self) -> Vec<String> {
        self.ztype_names.clone()
    }

    /// The gamete-type name directory (copy).
    #[getter]
    fn gtype_names(&self) -> Vec<String> {
        self.gtype_names.clone()
    }

    /// Adult age indices (copy).
    #[getter]
    fn adult_ages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_slice(py, &self.adult_ages))
    }
}

impl Blueprint {
    /// Extract a blueprint from the Python contract object.
    ///
    /// Reads the ``natal.contracts.Blueprint`` NamedTuple attributes once
    /// and owns plain Rust copies.  The Python tuple itself stays untouched.
    ///
    /// ## Parameters
    /// - `obj`: A ``natal.contracts.Blueprint`` instance.
    ///
    /// ## Returns
    /// An owned [`Blueprint`] mirroring the contract.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when attributes are missing or of the
    /// wrong type.
    pub fn from_python(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let adult_ages: Vec<i64> = extract_i64_vec(obj, "adult_ages")?;
        Ok(Self {
            n_sexes: extract_i64(obj, "n_sexes")? as usize,
            n_ages: extract_i64(obj, "n_ages")? as usize,
            n_ztypes: extract_i64(obj, "n_ztypes")? as usize,
            n_gtypes: extract_i64(obj, "n_gtypes")? as usize,
            n_glabs: extract_i64(obj, "n_glabs")? as usize,
            new_adult_age: extract_i64(obj, "new_adult_age")? as usize,
            adult_ages,
            stochastic: extract_bool(obj, "stochastic")?,
            continuous_sampling: extract_bool(obj, "continuous_sampling")?,
            fixed_egg_count: extract_bool(obj, "fixed_egg_count")?,
            has_sex_chromosomes: extract_bool(obj, "has_sex_chromosomes")?,
            extreme_speed_mode: extract_i64(obj, "extreme_speed_mode")?,
            ztype_names: extract_string_vec(obj, "ztype_names")?,
            gtype_names: extract_string_vec(obj, "gtype_names")?,
            female_only_by_sex_chrom: extract_bool_vec(obj, "female_only_by_sex_chrom")?,
            male_only_by_sex_chrom: extract_bool_vec(obj, "male_only_by_sex_chrom")?,
            initial_individual_count: extract_f64_vec(obj, "initial_individual_count")?,
            initial_sperm_storage: extract_f64_vec(obj, "initial_sperm_storage")?,
            n_demes: extract_i64(obj, "n_demes")? as usize,
            migration_indptr: extract_i64_vec(obj, "migration_indptr")?,
            migration_dest_idx: extract_i64_vec(obj, "migration_dest_idx")?,
            migration_weights: extract_f64_vec(obj, "migration_weights")?,
        })
    }
}
