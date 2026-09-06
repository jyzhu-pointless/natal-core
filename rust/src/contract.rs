//! The contract structs handed across the Python/Rust boundary.
//!
//! [`Blueprint`] is the frozen model specification (rebuild to change),
//! [`Params`] is every runtime-mutable *ecology* value stored as per-deme
//! columns, and [`TensorSet`] is the genotype-indexed *genetics* section.
//!
//! Slice-5 stage-2 layout (structure-of-arrays across demes):
//!
//! - Every ecology scalar is a ``Vec<f64>`` column of length ``n_demes``
//!   (a panmictic model is one deme, so its columns have length 1 and the
//!   kernels read entry 0 — the flat contents of a length-1 column are
//!   bit-identical to the pre-columnization scalars).
//! - Every ecology vector carries a leading deme dimension
//!   (``(n_demes, ...)`` flattened row-major).
//! - The eight genetics tables live in [`TensorSet`], which demes *share*:
//!   a spatial session owns one ecology column set plus a bank of
//!   [`TensorSet`] variants indexed per deme, so heterogeneous genetics
//!   never clone ecology and identical genetics never clone tensors.
//!
//! A session is built once from plain contract data (``from_parts``), then
//! params mutate *in place* through [`Params::apply`] / [`Params::tensor_write`]
//! / [`Params::pull_fields`] — no fingerprints, no session rebuilds, no RNG
//! resets.
//!
//! Field names and shapes mirror the Python contracts in
//! ``natal/contracts/{blueprint,params}.py`` (CONTRACTS_VERSION >= 2).

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

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
    // -- spatial domain (slice 5) --
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

/// Contract names of the eight genetics tables carried by [`TensorSet`].
pub const GENETICS_TENSORS: [&str; 8] = [
    "viability_fitness",
    "fecundity_fitness",
    "sexual_selection_fitness",
    "zygote_viability_fitness",
    "offspring_tensor",
    "meiosis_map",
    "female_ztype_compatibility",
    "male_ztype_compatibility",
];

/// Whether a contract field name denotes a genetics (genotype-indexed)
/// tensor.  Genetics tensors live in [`TensorSet`] instead of [`Params`].
#[must_use]
pub fn is_genetics_tensor(name: &str) -> bool {
    GENETICS_TENSORS.contains(&name)
}

/// The genetics section: eight genotype-indexed tables (flat row-major).
///
/// Demes with identical genetics share one ``TensorSet`` through the
/// spatial session's variant bank, so heterogeneous models pay for the
/// tables once per *genetics variant*, not once per deme.
#[derive(Clone, Default)]
pub struct TensorSet {
    pub viability_fitness: Vec<f64>,          // (2, A, Z)
    pub fecundity_fitness: Vec<f64>,          // (2, Z)
    pub sexual_selection_fitness: Vec<f64>,   // (Z, Z)
    pub zygote_viability_fitness: Vec<f64>,   // (2, Z)
    pub offspring_tensor: Vec<f64>,           // (Z, Z, Z)
    pub meiosis_map: Vec<f64>,                // (2, Z, G)
    pub female_ztype_compatibility: Vec<f64>, // (Z,)
    pub male_ztype_compatibility: Vec<f64>,   // (Z,)
}

impl TensorSet {
    /// Extract the eight genetics tables from the Python contract object.
    ///
    /// ## Parameters
    /// - `obj`: A ``natal.contracts.Params`` instance.
    ///
    /// ## Returns
    /// An owned [`TensorSet`] mirroring the contract tables.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when an attribute is missing or of the
    /// wrong type.
    pub fn from_python(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            viability_fitness: extract_f64_vec(obj, "viability_fitness")?,
            fecundity_fitness: extract_f64_vec(obj, "fecundity_fitness")?,
            sexual_selection_fitness: extract_f64_vec(obj, "sexual_selection_fitness")?,
            zygote_viability_fitness: extract_f64_vec(obj, "zygote_viability_fitness")?,
            offspring_tensor: extract_f64_vec(obj, "offspring_tensor")?,
            meiosis_map: extract_f64_vec(obj, "meiosis_map")?,
            female_ztype_compatibility: extract_f64_vec(obj, "female_ztype_compatibility")?,
            male_ztype_compatibility: extract_f64_vec(obj, "male_ztype_compatibility")?,
        })
    }

    /// Extract the eight genetics tables from a ``{name: 1-D array}``
    /// mapping (the Python-side variant bank hands flat tensors over).
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when a name is unknown or a value is not a
    /// float64 array.
    pub fn from_dict(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = obj.downcast::<PyDict>()?;
        let mut set = Self::default();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            let values: Vec<f64> = value
                .extract::<PyReadonlyArrayDyn<'_, f64>>()?
                .as_slice()?
                .to_vec();
            match set.tensor_field_len(&key) {
                Ok(_) => *set.tensor_mut(&key)? = values,
                Err(_) => {
                    return Err(PyKeyError::new_err(format!(
                        "unknown genetics tensor {key:?}"
                    )))
                }
            }
        }
        Ok(set)
    }

    /// Current length of one genetics tensor (name check for from_dict).
    fn tensor_field_len(&self, name: &str) -> PyResult<usize> {
        Ok(self.field_slice(name)?.len())
    }

    /// Validate table sizes against the blueprint dimensions.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on the first size mismatch.
    pub fn validate(&self, bp: &Blueprint) -> PyResult<()> {
        for name in GENETICS_TENSORS {
            let expected = Self::expected_len(bp, name)?;
            let got = self.stored_len(name)?;
            if got != expected {
                return Err(PyValueError::new_err(format!(
                    "genetics {name}: expected {expected} elements, got {got}"
                )));
            }
        }
        Ok(())
    }

    fn stored_len(&self, name: &str) -> PyResult<usize> {
        Ok(self.field_slice(name)?.len())
    }

    fn field_slice(&self, name: &str) -> PyResult<&[f64]> {
        Ok(match name {
            "viability_fitness" => &self.viability_fitness,
            "fecundity_fitness" => &self.fecundity_fitness,
            "sexual_selection_fitness" => &self.sexual_selection_fitness,
            "zygote_viability_fitness" => &self.zygote_viability_fitness,
            "offspring_tensor" => &self.offspring_tensor,
            "meiosis_map" => &self.meiosis_map,
            "female_ztype_compatibility" => &self.female_ztype_compatibility,
            "male_ztype_compatibility" => &self.male_ztype_compatibility,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown genetics tensor {other:?}"
                )))
            }
        })
    }

    /// Expected flat length of a genetics table, derived from the blueprint.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown names.
    pub fn expected_len(bp: &Blueprint, name: &str) -> PyResult<usize> {
        let (a, z, g) = (bp.n_ages, bp.n_ztypes, bp.n_gtypes);
        Ok(match name {
            "viability_fitness" => 2 * a * z,
            "fecundity_fitness" | "zygote_viability_fitness" => 2 * z,
            "sexual_selection_fitness" => z * z,
            "offspring_tensor" => z * z * z,
            "meiosis_map" => 2 * z * g,
            "female_ztype_compatibility" | "male_ztype_compatibility" => z,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown genetics tensor {other:?}"
                )))
            }
        })
    }

    /// Mutable access to one genetics tensor by contract name.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown names.
    fn tensor_mut(&mut self, name: &str) -> PyResult<&mut Vec<f64>> {
        Ok(match name {
            "viability_fitness" => &mut self.viability_fitness,
            "fecundity_fitness" => &mut self.fecundity_fitness,
            "sexual_selection_fitness" => &mut self.sexual_selection_fitness,
            "zygote_viability_fitness" => &mut self.zygote_viability_fitness,
            "offspring_tensor" => &mut self.offspring_tensor,
            "meiosis_map" => &mut self.meiosis_map,
            "female_ztype_compatibility" => &mut self.female_ztype_compatibility,
            "male_ztype_compatibility" => &mut self.male_ztype_compatibility,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown genetics tensor {other:?}"
                )))
            }
        })
    }

    /// Pull exactly the named genetics tensors from the Python contract
    /// object (validated, atomic per call).
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for non-genetics names and ``PyValueError``
    /// on size mismatch; nothing is written when any field fails.
    pub fn pull_fields(
        &mut self,
        bp: &Blueprint,
        fields: &[String],
        source: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pending: Vec<(String, Vec<f64>)> = Vec::with_capacity(fields.len());
        for field in fields {
            if !is_genetics_tensor(field) {
                return Err(PyKeyError::new_err(format!(
                    "{field:?} is not a genetics tensor"
                )));
            }
            let expected = Self::expected_len(bp, field)?;
            let array = source
                .getattr(field.as_str())?
                .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
            if array.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "genetics {field}: expected {expected} elements, got {}",
                    array.len()
                )));
            }
            pending.push((field.clone(), extract_f64_vec(source, field)?));
        }
        for (field, values) in pending {
            *self.tensor_mut(&field)? = values;
        }
        Ok(())
    }

    /// Whole-tensor contents write: validate name and size, then copy in.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown names and ``PyValueError`` on
    /// size mismatch; the previous contents are preserved on failure.
    pub fn tensor_write(&mut self, bp: &Blueprint, name: &str, values: Vec<f64>) -> PyResult<()> {
        let expected = Self::expected_len(bp, name)?;
        if values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "genetics {name}: expected {expected} elements, got {}",
                values.len()
            )));
        }
        *self.tensor_mut(name)? = values;
        Ok(())
    }

    /// Read a copy of a genetics tensor.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown names.
    pub fn get_tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_slice(py, self.field_slice(name)?))
    }
}

/// Extract an int scalar, accepting Python ints and 0-d NumPy arrays.
fn extract_i64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<i64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<i64>()
}

/// Extract a float scalar, accepting Python floats and 0-d NumPy arrays.
fn extract_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<f64>()
}

/// Extract a bool scalar.
fn extract_bool(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    obj.getattr(name)?.extract::<bool>()
}

/// Extract a flat ``f64`` copy of any float64 NumPy array attribute.
///
/// Dimension-agnostic: blueprint/params arrays range from 1-D vectors to
/// the 3-D initial population; all are copied in row-major order.
fn extract_f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let array = obj
        .getattr(name)?
        .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a flat ``i64`` copy of an int64 NumPy array attribute.
fn extract_i64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let array = obj
        .getattr(name)?
        .extract::<PyReadonlyArrayDyn<'_, i64>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a flat ``bool`` copy of a bool NumPy array attribute.
fn extract_bool_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    let array = obj
        .getattr(name)?
        .extract::<PyReadonlyArrayDyn<'_, bool>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a tuple/sequence of strings attribute.
fn extract_string_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
    obj.getattr(name)?.extract::<Vec<String>>()
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

/// Whether a contract field name denotes an ecology tensor (vector) field
/// owned by [`Params`].  Genetics tensors are *not* ecology tensors.
fn is_ecology_tensor(name: &str) -> bool {
    matches!(name, |"survival_rates"| "mating_rates"
        | "reproduction_rates"
        | "fertility"
        | "competition_weights"
        | "equilibrium_distribution"
        | "migration_rate")
}

/// Extract scalar custom slots (bool/int/float) from the contract object.
///
/// Array-valued custom slots are deliberately skipped: no Rust kernel
/// consumes them and the Python side keeps ownership of the arrays.
fn extract_custom_slots(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, f64>> {
    let dict = obj.getattr("custom_slots")?;
    let dict = dict.downcast::<PyDict>()?;
    let mut slots = HashMap::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        if let Ok(number) = value.extract::<f64>() {
            slots.insert(key, number);
        }
    }
    Ok(slots)
}

/// Every runtime-mutable *ecology* value, stored as per-deme columns.
///
/// Column layout (slice-5 stage 2): scalar columns have length ``n_demes``
/// and vector columns are ``(n_demes, ...)`` flattened row-major.  A
/// panmictic model owns one deme, so its columns have length 1 and every
/// kernel reads entry 0 — bit-identical to the pre-columnization layout.
///
/// The genetics section lives in [`TensorSet`] (shared across demes via
/// the session's variant bank); all writes go through the validated
/// channels ([`Params::apply`], [`Params::tensor_write`],
/// [`Params::pull_fields`]).
#[derive(Clone)]
pub struct Params {
    /// Number of deme columns carried by every field.
    pub n_demes: usize,
    // -- ecology: per-deme scalar columns (length n_demes) --
    pub carrying_capacity: Vec<f64>,
    pub eggs_per_female: Vec<f64>,
    pub sex_ratio: Vec<f64>,
    pub sperm_displacement_rate: Vec<f64>,
    pub low_density_growth_rate: Vec<f64>,
    pub growth_mode: Vec<i64>,
    pub external_expected_eggs: Vec<f64>,
    // -- ecology: per-deme vector columns (leading n_demes dim) --
    pub survival_rates: Vec<f64>,           // (n_demes, 2, A)
    pub mating_rates: Vec<f64>,             // (n_demes, 2, A)
    pub reproduction_rates: Vec<f64>,       // (n_demes, A)
    pub fertility: Vec<f64>,                // (n_demes, A)
    pub competition_weights: Vec<f64>,      // (n_demes, A)
    pub equilibrium_distribution: Vec<f64>, // (n_demes, 2, A) or empty (derive mode)
    pub migration_rate: Vec<f64>,           // (n_demes, 2, A) or empty (not declared)
    // -- custom slots --
    pub custom_slots: HashMap<String, f64>,
}

/// Ecology scalar column names (f64 channels).
const ECOLOGY_SCALAR_COLUMNS: [&str; 6] = [
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
    "external_expected_eggs",
];

/// Canonical ``Op.set_param`` id order — generated from
/// ``src/natal/parameters.jsonc`` via ``scripts/generate_param_tables.py``
/// in the fixed ``ECO_PARAM_NAMES`` wire order.  The interpreter indexes
/// its ``eco_values`` slice by position here; the order is a wire contract.
pub use crate::eco_param_wire::ECO_PARAM_COLUMNS;

/// Ecology vector column names (columnized tensor channels).
const ECOLOGY_TENSOR_COLUMNS: [&str; 7] = [
    "survival_rates",
    "mating_rates",
    "reproduction_rates",
    "fertility",
    "competition_weights",
    "equilibrium_distribution",
    "migration_rate",
];

/// A mutable scalar write target.
enum ScalarRef<'a> {
    F64(&'a mut f64),
    I64(&'a mut i64),
}

impl Params {
    /// Tile one per-deme value into a column of length ``n_demes``.
    #[cfg(test)]
    fn tile<T: Copy>(value: T, n_demes: usize) -> Vec<T> {
        vec![value; n_demes]
    }

    /// Tile a per-deme vector into a columnized ``(n_demes, ...)`` vector.
    pub(crate) fn tile_vec(values: &[f64], n_demes: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(values.len() * n_demes);
        for _ in 0..n_demes {
            out.extend_from_slice(values);
        }
        out
    }

    /// Cut one deme's segment out of a columnized vector field.
    ///
    /// Empty sentinel columns (derive-mode equilibrium, undeclared
    /// migration) stay empty; the per-deme extent is inferred from the
    /// deme count.
    fn cut_deme_segment(column: &[f64], n_demes: usize, deme: usize) -> Vec<f64> {
        if column.is_empty() || n_demes == 0 {
            return Vec::new();
        }
        let per = column.len() / n_demes;
        column[deme * per..(deme + 1) * per].to_vec()
    }

    /// Extract one deme's ecology into a fresh single-deme ``Params``.
    ///
    /// Spatial parallel ticks need a per-deme mutable write target for
    /// ``Op.set_param`` commits, but the session's ecology columns cannot
    /// be borrowed ``&mut`` by several demes at once.  Each deme therefore
    /// ticks against this private single-column copy; ``assemble_deme``
    /// reads position 0, whose contents are exactly the values the source
    /// column held at this deme (so a fresh assembly is numerically
    /// identical to the session's per-deme config).  Genetics are *not*
    /// carried — the caller shares the immutable [`TensorSet`].
    ///
    /// ## Parameters
    /// - `deme`: Deme whose column entries and vector segments are copied.
    ///
    /// ## Returns
    /// A ``Params`` with ``n_demes == 1`` holding only that deme's ecology
    /// (scalar custom slots are shared by clone).
    ///
    /// ## Panics
    /// Panics when *deme* is out of range for the scalar columns.
    pub fn single_deme(&self, deme: usize) -> Params {
        Params {
            n_demes: 1,
            carrying_capacity: vec![self.carrying_capacity[deme]],
            eggs_per_female: vec![self.eggs_per_female[deme]],
            sex_ratio: vec![self.sex_ratio[deme]],
            sperm_displacement_rate: vec![self.sperm_displacement_rate[deme]],
            low_density_growth_rate: vec![self.low_density_growth_rate[deme]],
            growth_mode: vec![self.growth_mode[deme]],
            external_expected_eggs: vec![self.external_expected_eggs[deme]],
            survival_rates: Self::cut_deme_segment(&self.survival_rates, self.n_demes, deme),
            mating_rates: Self::cut_deme_segment(&self.mating_rates, self.n_demes, deme),
            reproduction_rates: Self::cut_deme_segment(
                &self.reproduction_rates,
                self.n_demes,
                deme,
            ),
            fertility: Self::cut_deme_segment(&self.fertility, self.n_demes, deme),
            competition_weights: Self::cut_deme_segment(
                &self.competition_weights,
                self.n_demes,
                deme,
            ),
            equilibrium_distribution: Self::cut_deme_segment(
                &self.equilibrium_distribution,
                self.n_demes,
                deme,
            ),
            // The migration-rate column is cut like every other vector
            // column: a deme's lifecycle consumes only its own (2, A) rate
            // segment; the empty sentinel stays empty.
            migration_rate: Self::cut_deme_segment(&self.migration_rate, self.n_demes, deme),
            custom_slots: self.custom_slots.clone(),
        }
    }

    /// Extract every parameter from the Python contract object.
    ///
    /// The Python contract carries *per-deme-0* values (scalars plus
    /// ``(2, A)``-shaped vectors); they are tiled into ``n_demes`` columns.
    /// The migration-rate column is taken as-is because the contract
    /// already carries the full ``(n_demes, 2, A)`` extent (or the empty
    /// "not declared" sentinel).
    ///
    /// ## Parameters
    /// - `obj`: A ``natal.contracts.Params`` instance.
    /// - `n_demes`: Number of deme columns to produce.
    ///
    /// ## Returns
    /// An owned [`Params`] with columnized ecology.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when attributes are missing or of the wrong
    /// type, or ``PyTypeError`` for unsupported custom-slot values.
    pub fn from_python(obj: &Bound<'_, PyAny>, n_demes: usize) -> PyResult<Self> {
        let survival = extract_f64_vec(obj, "survival_rates")?;
        let mating = extract_f64_vec(obj, "mating_rates")?;
        let reproduction = extract_f64_vec(obj, "reproduction_rates")?;
        let fertility = extract_f64_vec(obj, "fertility")?;
        let competition = extract_f64_vec(obj, "competition_weights")?;
        let equilibrium = extract_f64_vec(obj, "equilibrium_distribution")?;
        Ok(Self {
            n_demes,
            carrying_capacity: vec![extract_f64(obj, "carrying_capacity")?; n_demes],
            eggs_per_female: vec![extract_f64(obj, "eggs_per_female")?; n_demes],
            sex_ratio: vec![extract_f64(obj, "sex_ratio")?; n_demes],
            sperm_displacement_rate: vec![extract_f64(obj, "sperm_displacement_rate")?; n_demes],
            low_density_growth_rate: vec![extract_f64(obj, "low_density_growth_rate")?; n_demes],
            growth_mode: vec![extract_i64(obj, "growth_mode")?; n_demes],
            external_expected_eggs: vec![extract_f64(obj, "external_expected_eggs")?; n_demes],
            survival_rates: Self::tile_vec(&survival, n_demes),
            mating_rates: Self::tile_vec(&mating, n_demes),
            reproduction_rates: Self::tile_vec(&reproduction, n_demes),
            fertility: Self::tile_vec(&fertility, n_demes),
            competition_weights: Self::tile_vec(&competition, n_demes),
            equilibrium_distribution: if equilibrium.is_empty() {
                Vec::new()
            } else {
                Self::tile_vec(&equilibrium, n_demes)
            },
            migration_rate: extract_f64_vec(obj, "migration_rate")?,
            custom_slots: extract_custom_slots(obj)?,
        })
    }

    /// Extract columnized ecology from a ``{field: 1-D array}`` mapping.
    ///
    /// This is the spatial-session boundary: the Python adapter gathers
    /// true per-deme values into flat columns (scalar columns length
    /// ``n_demes``, vector columns ``n_demes`` times their per-deme
    /// extent, ``growth_mode`` int64) and the session owns them without
    /// any tiling.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown fields and ``PyValueError`` when
    /// a column length disagrees with ``n_demes`` or a value is not a
    /// float64/int64 array.
    pub fn from_columns(obj: &Bound<'_, PyAny>, n_demes: usize) -> PyResult<Self> {
        let mut params = Self {
            n_demes,
            carrying_capacity: vec![0.0; n_demes],
            eggs_per_female: vec![0.0; n_demes],
            sex_ratio: vec![0.5; n_demes],
            sperm_displacement_rate: vec![0.0; n_demes],
            low_density_growth_rate: vec![0.0; n_demes],
            growth_mode: vec![0; n_demes],
            external_expected_eggs: vec![-1.0; n_demes],
            survival_rates: Vec::new(),
            mating_rates: Vec::new(),
            reproduction_rates: Vec::new(),
            fertility: Vec::new(),
            competition_weights: Vec::new(),
            equilibrium_distribution: Vec::new(),
            migration_rate: Vec::new(),
            custom_slots: HashMap::new(),
        };
        let dict = obj.downcast::<PyDict>()?;
        let mut seen = std::collections::HashSet::new();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            if !seen.insert(key.clone()) {
                return Err(PyValueError::new_err(format!(
                    "duplicate ecology column {key:?}"
                )));
            }
            if key == "growth_mode" {
                let values = value
                    .extract::<PyReadonlyArrayDyn<'_, i64>>()?
                    .as_slice()?
                    .to_vec();
                Self::expect_column_len(&key, values.len(), n_demes)?;
                params.growth_mode = values;
                continue;
            }
            let values: Vec<f64> = value
                .extract::<PyReadonlyArrayDyn<'_, f64>>()?
                .as_slice()?
                .to_vec();
            if ECOLOGY_SCALAR_COLUMNS.contains(&key.as_str()) {
                Self::expect_column_len(&key, values.len(), n_demes)?;
                let column = params.scalar_column_mut(&key)?;
                *column = values;
            } else if ECOLOGY_TENSOR_COLUMNS.contains(&key.as_str()) {
                let is_sentinel = key == "equilibrium_distribution" || key == "migration_rate";
                if values.is_empty() && is_sentinel {
                    continue;
                }
                // Per-deme extent is inferred here: a column must be
                // n_demes equal-length tiles.  Exact per-deme sizes are
                // validated against the blueprint by `validate` at
                // construction.
                if n_demes == 0 || values.len() % n_demes != 0 {
                    return Err(PyValueError::new_err(format!(
                        "ecology column {key}: length {} is not a multiple of {n_demes} demes",
                        values.len()
                    )));
                }
                match key.as_str() {
                    "survival_rates" => params.survival_rates = values,
                    "mating_rates" => params.mating_rates = values,
                    "reproduction_rates" => params.reproduction_rates = values,
                    "fertility" => params.fertility = values,
                    "competition_weights" => params.competition_weights = values,
                    "equilibrium_distribution" => params.equilibrium_distribution = values,
                    "migration_rate" => params.migration_rate = values,
                    _ => unreachable!("name matched the tensor column list"),
                }
            } else {
                return Err(PyKeyError::new_err(format!(
                    "unknown ecology column {key:?}"
                )));
            }
        }
        Ok(params)
    }

    /// Reject a scalar column whose length differs from the deme count.
    fn expect_column_len(key: &str, got: usize, n_demes: usize) -> PyResult<()> {
        if got != n_demes {
            return Err(PyValueError::new_err(format!(
                "ecology column {key}: expected {n_demes} entries, got {got}"
            )));
        }
        Ok(())
    }

    /// Mutable access to one f64 scalar column by name.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown column names.
    fn scalar_column_mut(&mut self, name: &str) -> PyResult<&mut Vec<f64>> {
        Ok(match name {
            "carrying_capacity" => &mut self.carrying_capacity,
            "eggs_per_female" => &mut self.eggs_per_female,
            "sex_ratio" => &mut self.sex_ratio,
            "sperm_displacement_rate" => &mut self.sperm_displacement_rate,
            "low_density_growth_rate" => &mut self.low_density_growth_rate,
            "external_expected_eggs" => &mut self.external_expected_eggs,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown scalar params field {other:?}"
                )))
            }
        })
    }

    /// Read one canonical ECO param id at a deme (``Op.set_param`` table).
    ///
    /// ## Parameters
    /// - `id`: Index into [`ECO_PARAM_COLUMNS`].
    /// - `deme`: Deme column to read.
    ///
    /// ## Returns
    /// The column value, or 0.0 when out of range (defensive: ids come
    /// from the validated CSR program).
    pub fn eco_value(&self, id: usize, deme: usize) -> f64 {
        let name = ECO_PARAM_COLUMNS[id.min(ECO_PARAM_COLUMNS.len() - 1)];
        let column = match name {
            "carrying_capacity" => &self.carrying_capacity,
            "eggs_per_female" => &self.eggs_per_female,
            "sex_ratio" => &self.sex_ratio,
            "sperm_displacement_rate" => &self.sperm_displacement_rate,
            _ => &self.low_density_growth_rate,
        };
        column.get(deme).copied().unwrap_or(0.0)
    }

    /// Write one canonical ECO param id at a deme (``Op.set_param`` table).
    ///
    /// ## Parameters
    /// - `id`: Index into [`ECO_PARAM_COLUMNS`].
    /// - `deme`: Deme column to write.
    /// - `value`: New value.
    pub fn set_eco_value(&mut self, id: usize, deme: usize, value: f64) {
        let name = ECO_PARAM_COLUMNS[id.min(ECO_PARAM_COLUMNS.len() - 1)];
        let column = match name {
            "carrying_capacity" => &mut self.carrying_capacity,
            "eggs_per_female" => &mut self.eggs_per_female,
            "sex_ratio" => &mut self.sex_ratio,
            "sperm_displacement_rate" => &mut self.sperm_displacement_rate,
            _ => &mut self.low_density_growth_rate,
        };
        if let Some(slot) = column.get_mut(deme) {
            *slot = value;
        }
    }

    /// Resolve a scalar field at one deme to a mutable write target.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown or non-scalar fields and
    /// ``PyValueError`` when the deme index is out of range.
    fn scalar_ref(&mut self, name: &str, deme: usize) -> PyResult<ScalarRef<'_>> {
        if deme >= self.n_demes {
            return Err(PyValueError::new_err(format!(
                "deme {deme} out of range for {} ecology columns",
                self.n_demes
            )));
        }
        Ok(match name {
            "carrying_capacity" => ScalarRef::F64(&mut self.carrying_capacity[deme]),
            "eggs_per_female" => ScalarRef::F64(&mut self.eggs_per_female[deme]),
            "sex_ratio" => ScalarRef::F64(&mut self.sex_ratio[deme]),
            "sperm_displacement_rate" => ScalarRef::F64(&mut self.sperm_displacement_rate[deme]),
            "low_density_growth_rate" => ScalarRef::F64(&mut self.low_density_growth_rate[deme]),
            "growth_mode" => ScalarRef::I64(&mut self.growth_mode[deme]),
            "external_expected_eggs" => ScalarRef::F64(&mut self.external_expected_eggs[deme]),
            other => {
                if is_ecology_tensor(other) || is_genetics_tensor(other) {
                    return Err(PyValueError::new_err(format!(
                        "{other:?} is a tensor field; use tensor_write"
                    )));
                }
                return Err(PyKeyError::new_err(format!(
                    "unknown params field {other:?}"
                )));
            }
        })
    }

    /// Per-deme flat length of an ecology vector field.
    fn per_deme_len(&self, bp: &Blueprint, name: &str) -> PyResult<usize> {
        let a = bp.n_ages;
        Ok(match name {
            "survival_rates" | "mating_rates" => 2 * a,
            "reproduction_rates" | "fertility" | "competition_weights" => a,
            "equilibrium_distribution" | "migration_rate" => 2 * a,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown params field {other:?}"
                )))
            }
        })
    }

    /// Expected flat length of a vector field (all demes).
    fn expected_len(&self, bp: &Blueprint, name: &str) -> PyResult<usize> {
        Ok(self.per_deme_len(bp, name)? * self.n_demes)
    }

    /// Copy a per-deme segment into a columnized tensor (validated).
    ///
    /// A derive-mode / not-declared sentinel column of a single-deme
    /// (panmictic) params adopts the written contents wholesale, preserving
    /// the pre-columnization widen semantics.  Multi-deme sentinel columns
    /// cannot be written one deme at a time and are rejected.
    fn write_deme_tensor(
        &mut self,
        bp: &Blueprint,
        name: &str,
        deme: usize,
        values: &[f64],
    ) -> PyResult<()> {
        if is_genetics_tensor(name) {
            return Err(PyKeyError::new_err(format!(
                "{name:?} is a genetics tensor; route it to the session genetics bank"
            )));
        }
        if !is_ecology_tensor(name) {
            return Err(PyKeyError::new_err(format!(
                "unknown params field {name:?}"
            )));
        }
        let per = self.per_deme_len(bp, name)?;
        if values.len() != per {
            return Err(PyValueError::new_err(format!(
                "Params.{name}: expected {per} elements for one deme, got {}",
                values.len()
            )));
        }
        let total = per * self.n_demes;
        let column: &mut Vec<f64> = match name {
            "survival_rates" => &mut self.survival_rates,
            "mating_rates" => &mut self.mating_rates,
            "reproduction_rates" => &mut self.reproduction_rates,
            "fertility" => &mut self.fertility,
            "competition_weights" => &mut self.competition_weights,
            "equilibrium_distribution" => &mut self.equilibrium_distribution,
            "migration_rate" => &mut self.migration_rate,
            _ => unreachable!("name matched the ecology tensor list"),
        };
        if column.len() == total {
            let offset = deme * per;
            column[offset..offset + per].copy_from_slice(values);
        } else if column.is_empty() && self.n_demes == 1 {
            // Panmictic widen: the sentinel column adopts declared contents.
            *column = values.to_vec();
        } else {
            return Err(PyValueError::new_err(format!(
                "Params.{name}: column holds {} elements, expected {total}; \
                 cannot write one deme of a sentinel column",
                column.len()
            )));
        }
        Ok(())
    }

    /// Pull exactly the named fields from the Python contract object.
    ///
    /// This is the directed refresh channel: only the listed scalars and
    /// tensors are read from *source* and written into ``self`` (ecology at
    /// the given deme) and into ``genetics`` when provided.  Unknown names
    /// fail before any write (method-level atomicity), and tensor pulls are
    /// size-checked against the blueprint.
    ///
    /// ## Parameters
    /// - `bp`: Blueprint providing the expected tensor sizes.
    /// - `deme`: Deme column the ecology writes target.
    /// - `fields`: Contract field names to pull.
    /// - `source`: A ``natal.contracts.Params`` instance carrying the
    ///   current values.
    /// - `genetics`: Optional genetics sink; required when any requested
    ///   field is a genetics tensor.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown fields and ``PyValueError`` on
    /// size mismatch; nothing is written when any field fails.
    pub fn pull_fields(
        &mut self,
        bp: &Blueprint,
        deme: usize,
        fields: &[String],
        source: &Bound<'_, PyAny>,
        genetics: Option<&mut TensorSet>,
    ) -> PyResult<()> {
        // Validate all names and sizes first so a failure leaves the owned
        // params untouched.  "custom_slots" is a legal request that resolves
        // to nothing on the Rust side (the scalar slots live in the Python
        // contract object; they reach the session via apply).
        let mut wants_custom_slots = false;
        let mut pending_scalars: Vec<(String, f64)> = Vec::with_capacity(fields.len());
        let mut pending_tensors: Vec<(String, Vec<f64>)> = Vec::with_capacity(fields.len());
        let mut pending_genetics: Vec<(String, Vec<f64>)> = Vec::with_capacity(fields.len());
        for field in fields {
            if field == "custom_slots" {
                wants_custom_slots = true;
                continue;
            }
            if is_genetics_tensor(field) {
                if genetics.is_none() {
                    return Err(PyKeyError::new_err(format!(
                        "{field:?} is a genetics tensor; this channel carries ecology only"
                    )));
                }
                let expected = TensorSet::expected_len(bp, field)?;
                let array = source
                    .getattr(field.as_str())?
                    .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
                if array.len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "genetics {field}: expected {expected} elements, got {}",
                        array.len()
                    )));
                }
                pending_genetics.push((field.clone(), extract_f64_vec(source, field)?));
                continue;
            }
            // Scalar names must resolve through the scalar channel; tensor
            // names through the deme-segment channel.  Unknown names fail
            // here before any write happens.
            if is_ecology_tensor(field) {
                let array = source
                    .getattr(field.as_str())?
                    .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
                let expected = self.per_deme_len(bp, field)?;
                let is_empty_sentinel = (field == "equilibrium_distribution"
                    || field == "migration_rate")
                    && array.len() == 0;
                if array.len() != expected && !is_empty_sentinel {
                    return Err(PyValueError::new_err(format!(
                        "Params.{field}: expected {expected} elements, got {}",
                        array.len()
                    )));
                }
                pending_tensors.push((field.clone(), extract_f64_vec(source, field)?));
            } else {
                self.scalar_ref(field, deme)?;
                pending_scalars.push((field.clone(), extract_f64(source, field)?));
            }
        }
        // Commit pass: scalars first, then tensors, then genetics.
        for (field, value) in &pending_scalars {
            match self.scalar_ref(field, deme)? {
                ScalarRef::F64(target) => *target = *value,
                ScalarRef::I64(target) => *target = *value as i64,
            }
        }
        for (field, values) in &pending_tensors {
            if values.is_empty() {
                // Empty sentinel pull on a derive/not-declared column:
                // nothing to copy into the deme segment.
                continue;
            }
            self.write_deme_tensor(bp, field, deme, values)?;
        }
        if let Some(sink) = genetics {
            for (field, values) in &pending_genetics {
                sink.tensor_write(bp, field, values.clone())?;
            }
        }
        if wants_custom_slots {
            self.custom_slots = extract_custom_slots(source)?;
        }
        Ok(())
    }

    /// Read-only lookup of a vector field's current length.
    fn stored_len(&self, name: &str) -> PyResult<usize> {
        let len = match name {
            "survival_rates" => self.survival_rates.len(),
            "mating_rates" => self.mating_rates.len(),
            "reproduction_rates" => self.reproduction_rates.len(),
            "fertility" => self.fertility.len(),
            "competition_weights" => self.competition_weights.len(),
            "equilibrium_distribution" => self.equilibrium_distribution.len(),
            "migration_rate" => self.migration_rate.len(),
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown params field {other:?}"
                )))
            }
        };
        Ok(len)
    }
}

impl Params {
    /// Validate that every column and vector field matches the
    /// blueprint-derived size.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on the first size mismatch.
    pub fn validate(&self, bp: &Blueprint) -> PyResult<()> {
        // Scalar columns carry exactly one entry per deme.
        let scalar_lens: [(&str, usize); 7] = [
            ("carrying_capacity", self.carrying_capacity.len()),
            ("eggs_per_female", self.eggs_per_female.len()),
            ("sex_ratio", self.sex_ratio.len()),
            (
                "sperm_displacement_rate",
                self.sperm_displacement_rate.len(),
            ),
            (
                "low_density_growth_rate",
                self.low_density_growth_rate.len(),
            ),
            ("growth_mode", self.growth_mode.len()),
            ("external_expected_eggs", self.external_expected_eggs.len()),
        ];
        for (name, got) in scalar_lens {
            if got != self.n_demes {
                return Err(PyValueError::new_err(format!(
                    "Params.{name}: expected {} column entries, got {got}",
                    self.n_demes
                )));
            }
        }
        // Vector columns carry n_demes per-deme extents.
        for name in ECOLOGY_TENSOR_COLUMNS {
            let got = self.stored_len(name)?;
            let expected = self.expected_len(bp, name)?;
            let is_empty_sentinel =
                (name == "equilibrium_distribution" || name == "migration_rate") && got == 0;
            if got != expected && !is_empty_sentinel {
                return Err(PyValueError::new_err(format!(
                    "Params.{name}: expected {expected} elements, got {got}"
                )));
            }
        }
        Ok(())
    }

    /// Batch scalar write: validate every entry, then commit atomically.
    ///
    /// Writes target deme 0 (the panmictic/homogeneous write channel).
    /// Accepts both float and int scalars (the growth-mode selector is an
    /// int channel); the value is written through its native channel.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown fields and ``PyValueError`` for
    /// tensor fields; nothing is written when any entry fails.
    pub fn apply(&mut self, writes: HashMap<String, f64>) -> PyResult<()> {
        // Validate names AND field categories first (method-level atomicity:
        // nothing is written when any entry fails).
        for name in writes.keys() {
            self.scalar_ref(name, 0)?;
        }
        for (name, value) in writes {
            match self.scalar_ref(&name, 0)? {
                ScalarRef::F64(target) => *target = value,
                ScalarRef::I64(target) => *target = value as i64,
            }
        }
        Ok(())
    }

    /// Whole-tensor contents write: validate name and size, then copy in.
    ///
    /// Only ecology tensors live here; genetics tensors are rejected (the
    /// session routes them to its [`TensorSet`]).
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown or genetics fields and
    /// ``PyValueError`` on size mismatch; the previous contents are
    /// preserved on failure.
    pub fn tensor_write(&mut self, bp: &Blueprint, name: &str, values: Vec<f64>) -> PyResult<()> {
        if is_genetics_tensor(name) {
            return Err(PyKeyError::new_err(format!(
                "{name:?} is a genetics tensor; route it to the session genetics bank"
            )));
        }
        let expected = self.expected_len(bp, name)?;
        // The derive-mode sentinel for equilibrium_distribution (empty) is
        // legal content: writing full length widens it, writing empty keeps
        // it, and any other size is rejected.
        let keeps_empty_sentinel = matches!(name, "equilibrium_distribution" | "migration_rate")
            && {
                let current_empty = match name {
                    "equilibrium_distribution" => self.equilibrium_distribution.is_empty(),
                    _ => self.migration_rate.is_empty(),
                };
                current_empty && values.is_empty()
            };
        if values.len() != expected && !keeps_empty_sentinel {
            return Err(PyValueError::new_err(format!(
                "Params.{name}: expected {expected} elements, got {}",
                values.len()
            )));
        }
        match name {
            "survival_rates" => self.survival_rates = values,
            "mating_rates" => self.mating_rates = values,
            "reproduction_rates" => self.reproduction_rates = values,
            "fertility" => self.fertility = values,
            "competition_weights" => self.competition_weights = values,
            "equilibrium_distribution" => self.equilibrium_distribution = values,
            "migration_rate" => self.migration_rate = values,
            _ => {
                // Scalar names never reach here: expected_len raises
                // KeyError for them before any write happens.
                return Err(PyKeyError::new_err(format!(
                    "unknown or non-tensor params field {name:?}"
                )));
            }
        }
        Ok(())
    }

    /// Read a scalar field value (deme 0 column entry).
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown or non-scalar fields.
    pub fn get_scalar(&self, name: &str) -> PyResult<f64> {
        let deme0 = |column: &Vec<f64>| -> PyResult<f64> {
            column
                .first()
                .copied()
                .ok_or_else(|| PyValueError::new_err("params column is empty (n_demes == 0)"))
        };
        Ok(match name {
            "carrying_capacity" => deme0(&self.carrying_capacity)?,
            "eggs_per_female" => deme0(&self.eggs_per_female)?,
            "sex_ratio" => deme0(&self.sex_ratio)?,
            "sperm_displacement_rate" => deme0(&self.sperm_displacement_rate)?,
            "low_density_growth_rate" => deme0(&self.low_density_growth_rate)?,
            "growth_mode" => deme0_scalar_i64(&self.growth_mode)?,
            "external_expected_eggs" => deme0(&self.external_expected_eggs)?,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown or non-scalar params field {other:?}"
                )))
            }
        })
    }

    /// Read a copy of an ecology vector field (full column).
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown, genetics, or non-tensor fields.
    pub fn get_tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if is_genetics_tensor(name) {
            return Err(PyKeyError::new_err(format!(
                "{name:?} is a genetics tensor; read it from the session genetics bank"
            )));
        }
        let v: &Vec<f64> = match name {
            "survival_rates" => &self.survival_rates,
            "mating_rates" => &self.mating_rates,
            "reproduction_rates" => &self.reproduction_rates,
            "fertility" => &self.fertility,
            "competition_weights" => &self.competition_weights,
            "equilibrium_distribution" => &self.equilibrium_distribution,
            "migration_rate" => &self.migration_rate,
            other => {
                return Err(PyKeyError::new_err(format!(
                    "unknown or non-tensor params field {other:?}"
                )))
            }
        };
        Ok(PyArray1::from_slice(py, v))
    }
}

/// Deme-0 entry of an int64 column as f64 (growth-mode read channel).
fn deme0_scalar_i64(column: &[i64]) -> PyResult<f64> {
    column
        .first()
        .map(|&v| v as f64)
        .ok_or_else(|| PyValueError::new_err("params column is empty (n_demes == 0)"))
}

/// Session-level tensor read routing: genetics names read from the
/// [`TensorSet`], ecology names from [`Params`].
///
/// ## Errors
/// Returns ``PyKeyError`` for unknown names.
pub fn session_get_tensor<'py>(
    py: Python<'py>,
    params: &Params,
    genetics: &TensorSet,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if is_genetics_tensor(name) {
        genetics.get_tensor(py, name)
    } else {
        params.get_tensor(py, name)
    }
}

/// Session-level tensor write routing: genetics names write into the
/// [`TensorSet`], ecology names into [`Params`].
///
/// ## Errors
/// Returns ``PyKeyError`` for unknown names and ``PyValueError`` on size
/// mismatch; previous contents are preserved on failure.
pub fn session_tensor_write(
    bp: &Blueprint,
    params: &mut Params,
    genetics: &mut TensorSet,
    name: &str,
    values: Vec<f64>,
) -> PyResult<()> {
    if is_genetics_tensor(name) {
        genetics.tensor_write(bp, name, values)
    } else {
        params.tensor_write(bp, name, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal consistent pair (2 sexes, 2 ages, 2 ztypes, 2 gtypes).
    fn fixture() -> (Blueprint, Params, TensorSet) {
        let bp = Blueprint {
            n_sexes: 2,
            n_ages: 2,
            n_ztypes: 2,
            n_gtypes: 2,
            n_glabs: 1,
            new_adult_age: 1,
            adult_ages: vec![1],
            stochastic: false,
            continuous_sampling: false,
            fixed_egg_count: false,
            has_sex_chromosomes: false,
            extreme_speed_mode: 0,
            ztype_names: vec!["A|A".into(), "A|B".into()],
            gtype_names: vec!["A".into(), "B".into()],
            female_only_by_sex_chrom: vec![false, false],
            male_only_by_sex_chrom: vec![false, false],
            initial_individual_count: vec![0.0; 2 * 2 * 2],
            initial_sperm_storage: vec![],
            n_demes: 1,
            migration_indptr: vec![0, 0],
            migration_dest_idx: vec![],
            migration_weights: vec![],
        };
        let params = Params {
            n_demes: 1,
            carrying_capacity: vec![400.0],
            eggs_per_female: vec![30.0],
            sex_ratio: vec![0.5],
            sperm_displacement_rate: vec![0.1],
            low_density_growth_rate: vec![2.0],
            growth_mode: vec![2],
            external_expected_eggs: vec![-1.0],
            survival_rates: vec![0.9, 0.8, 0.85, 0.75],
            mating_rates: vec![0.9, 0.9, 0.8, 0.8],
            reproduction_rates: vec![0.0, 0.8],
            fertility: vec![0.0, 1.0],
            competition_weights: vec![1.0, 0.8],
            equilibrium_distribution: vec![],
            migration_rate: vec![],
            custom_slots: HashMap::new(),
        };
        let genetics = TensorSet {
            viability_fitness: vec![1.0; 8],
            fecundity_fitness: vec![1.0; 4],
            sexual_selection_fitness: vec![1.0; 4],
            zygote_viability_fitness: vec![1.0; 4],
            offspring_tensor: vec![0.0; 8],
            meiosis_map: vec![0.0; 8],
            female_ztype_compatibility: vec![0.5, 0.5],
            male_ztype_compatibility: vec![0.5, 0.5],
        };
        (bp, params, genetics)
    }

    /// An unknown field must fail in the validation pass: the legal entry in
    /// the same batch must not be committed (method-level atomicity).
    #[test]
    fn apply_unknown_field_writes_nothing() {
        let (_, mut params, _) = fixture();
        let writes = HashMap::from([
            ("carrying_capacity".to_string(), 111.0),
            ("no_such_field".to_string(), 1.0),
        ]);
        assert!(params.apply(writes).is_err());
        assert_eq!(params.carrying_capacity[0], 400.0);
    }

    /// A tensor field routed through the scalar channel must be rejected, and
    /// with a single entry there is nothing else to leak a partial write.
    #[test]
    fn apply_rejects_tensor_field_without_writing() {
        let (_, mut params, _) = fixture();
        let writes = HashMap::from([("survival_rates".to_string(), 0.5)]);
        assert!(params.apply(writes).is_err());
        assert_eq!(params.survival_rates, vec![0.9, 0.8, 0.85, 0.75]);
    }

    /// Scalars go through their native channels: f64 fields keep their float
    /// value, the growth-mode selector goes through the i64 channel.
    #[test]
    fn apply_writes_scalars_through_native_channels() {
        let (_, mut params, _) = fixture();
        let writes = HashMap::from([
            ("carrying_capacity".to_string(), 42.5),
            ("growth_mode".to_string(), 4.0),
        ]);
        assert!(params.apply(writes).is_ok());
        assert_eq!(params.carrying_capacity[0], 42.5);
        assert_eq!(params.growth_mode[0], 4);
        assert_eq!(params.get_scalar("carrying_capacity").unwrap(), 42.5);
        assert_eq!(params.get_scalar("growth_mode").unwrap(), 4.0);
    }

    /// Size must be validated before commit: a wrong-size write changes
    /// nothing, the correct size commits verbatim.
    #[test]
    fn tensor_write_rejects_wrong_size_and_preserves_contents() {
        let (bp, mut params, _) = fixture();
        assert!(params
            .tensor_write(&bp, "survival_rates", vec![0.1; 3])
            .is_err());
        assert_eq!(params.survival_rates, vec![0.9, 0.8, 0.85, 0.75]);
        assert!(params
            .tensor_write(&bp, "survival_rates", vec![0.1; 4])
            .is_ok());
        assert_eq!(params.survival_rates, vec![0.1, 0.1, 0.1, 0.1]);
    }

    /// Scalar fields are unreachable through the tensor channel, and
    /// genetics tensors are rejected by the ecology channel.
    #[test]
    fn tensor_write_rejects_scalar_and_genetics_fields() {
        let (bp, mut params, _) = fixture();
        assert!(params
            .tensor_write(&bp, "carrying_capacity", vec![1.0])
            .is_err());
        assert_eq!(params.carrying_capacity[0], 400.0);
        assert!(params
            .tensor_write(&bp, "viability_fitness", vec![1.0; 8])
            .is_err());
    }

    /// The derive-mode sentinel of ``equilibrium_distribution``: empty ->
    /// empty keeps deriving, empty -> full widens to declared contents, and
    /// every other transition (wrong size, full -> empty) is rejected with
    /// the previous contents preserved.
    #[test]
    fn tensor_write_equilibrium_empty_sentinel_semantics() {
        let (bp, mut params, _) = fixture();
        assert!(params
            .tensor_write(&bp, "equilibrium_distribution", vec![])
            .is_ok());
        assert!(params.equilibrium_distribution.is_empty());
        assert!(params
            .tensor_write(&bp, "equilibrium_distribution", vec![1.0, 2.0, 3.0, 4.0])
            .is_ok());
        assert_eq!(params.equilibrium_distribution, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(params
            .tensor_write(&bp, "equilibrium_distribution", vec![1.0; 3])
            .is_err());
        assert_eq!(params.equilibrium_distribution, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(params
            .tensor_write(&bp, "equilibrium_distribution", vec![])
            .is_err());
        assert_eq!(params.equilibrium_distribution, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// ``validate`` checks every column against the blueprint-derived size
    /// while accepting the empty equilibrium sentinel (derive mode).
    #[test]
    fn validate_enforces_blueprint_derived_sizes() {
        let (bp, mut params, genetics) = fixture();
        assert!(params.validate(&bp).is_ok());
        assert!(genetics.validate(&bp).is_ok());
        params.survival_rates = vec![0.1; 3]; // expected 1 deme * 2 * a = 4
        assert!(params.validate(&bp).is_err());
        let (bp2, fresh, _) = fixture();
        assert!(fresh.equilibrium_distribution.is_empty());
        assert!(fresh.validate(&bp2).is_ok());
    }

    /// Read channels reject cross-kind and unknown names.
    #[test]
    fn get_scalar_rejects_tensor_and_unknown_names() {
        let (_, params, _) = fixture();
        assert!(params.get_scalar("survival_rates").is_err());
        assert!(params.get_scalar("bogus").is_err());
    }

    /// Multi-deme columns: a per-deme scalar write lands only in the
    /// targeted deme's column entry.
    #[test]
    fn deme_scoped_scalar_writes_are_independent() {
        let (_, mut params, _) = fixture();
        params.n_demes = 3;
        params.carrying_capacity = vec![400.0, 400.0, 400.0];
        params.growth_mode = vec![2, 2, 2];
        let writes = HashMap::from([("carrying_capacity".to_string(), 50.0)]);
        assert!(params.apply(writes).is_ok());
        assert_eq!(params.carrying_capacity, vec![50.0, 400.0, 400.0]);
        // Direct deme-scoped scalar_ref write.
        match params.scalar_ref("carrying_capacity", 2).unwrap() {
            ScalarRef::F64(target) => *target = 7.0,
            ScalarRef::I64(_) => panic!("expected f64 slot"),
        }
        assert_eq!(params.carrying_capacity, vec![50.0, 400.0, 7.0]);
    }

    /// Columnized tensor writes at a deme segment leave other demes intact.
    #[test]
    fn deme_scoped_tensor_segments_are_isolated() {
        let (bp, mut params, _) = fixture();
        params.n_demes = 2;
        params.survival_rates = vec![0.9, 0.8, 0.85, 0.75, 0.9, 0.8, 0.85, 0.75];
        // Simulate the per-deme pull channel at deme 1: expected segment is
        // 2 * a = 4 elements, committed at offset 4.
        let segment = vec![0.2, 0.3, 0.4, 0.5];
        params
            .write_deme_tensor(&bp, "survival_rates", 1, &segment)
            .unwrap();
        assert_eq!(params.survival_rates[..4], vec![0.9, 0.8, 0.85, 0.75]);
        assert_eq!(params.survival_rates[4..], segment[..]);
        // Whole-column write still validates against n_demes * per_deme.
        assert!(params
            .tensor_write(&bp, "survival_rates", vec![0.1; 4])
            .is_err());
    }

    /// `from_python` tiles per-deme-0 values into n_demes columns while the
    /// migration-rate column is taken as-is.
    #[test]
    fn tile_semantics_keep_panmictic_bit_identical() {
        // Length-1 columns reproduce the flat pre-columnization layout.
        let (_, params, _) = fixture();
        assert_eq!(params.n_demes, 1);
        assert_eq!(params.survival_rates.len(), 4);
        // Tiling to 3 demes triples the vector; scalars repeat.
        let mut tiled = params.clone();
        tiled.n_demes = 3;
        tiled.carrying_capacity = Params::tile(400.0, 3);
        tiled.survival_rates = Params::tile_vec(&params.survival_rates, 3);
        assert_eq!(tiled.carrying_capacity, vec![400.0, 400.0, 400.0]);
        assert_eq!(tiled.survival_rates.len(), 12);
        assert_eq!(&tiled.survival_rates[..4], &params.survival_rates[..]);
    }

    /// Genetics writes through the dedicated channel are size-validated and
    /// atomic.
    #[test]
    fn genetics_tensor_write_validates_sizes() {
        let (bp, _, mut genetics) = fixture();
        assert!(genetics
            .tensor_write(&bp, "viability_fitness", vec![1.0; 7])
            .is_err());
        assert_eq!(genetics.viability_fitness, vec![1.0; 8]);
        assert!(genetics
            .tensor_write(&bp, "viability_fitness", vec![0.5; 8])
            .is_ok());
        assert_eq!(genetics.viability_fitness, vec![0.5; 8]);
        assert!(genetics
            .tensor_write(&bp, "survival_rates", vec![1.0])
            .is_err());
        assert!(TensorSet::expected_len(&bp, "nope").is_err());
    }

    /// ``single_deme`` is the HB-1 correctness core: a config assembled from
    /// the local single-deme copy must be numerically identical to the
    /// session's per-deme config, otherwise the local-EcoCtx spatial tick
    /// would change demography without any set_param write.  Deme 1 carries
    /// deliberately distinct ecology so a mis-cut segment cannot pass.
    #[test]
    fn single_deme_local_assembly_matches_per_deme_config() {
        let (bp, mut params, genetics) = fixture();
        // Widen to 3 demes; give every deme distinct ecology columns.
        params.n_demes = 3;
        params.carrying_capacity = vec![400.0, 650.0, 310.0];
        params.eggs_per_female = vec![30.0, 22.5, 41.0];
        params.sex_ratio = vec![0.5, 0.6, 0.45];
        params.sperm_displacement_rate = vec![0.1, 0.2, 0.05];
        params.low_density_growth_rate = vec![2.0, 3.5, 1.25];
        params.growth_mode = vec![2, 1, 2];
        params.external_expected_eggs = vec![-1.0, 5000.0, -1.0];
        params.survival_rates = Params::tile_vec(&params.survival_rates, 3);
        params.mating_rates = Params::tile_vec(&params.mating_rates, 3);
        params.reproduction_rates = Params::tile_vec(&params.reproduction_rates, 3);
        params.fertility = Params::tile_vec(&params.fertility, 3);
        params.competition_weights = Params::tile_vec(&params.competition_weights, 3);
        // Skew deme 1's vector segments so a wrong cut is detectable.
        for i in 4..8 {
            params.survival_rates[i] *= 0.5;
            params.mating_rates[i] *= 0.5;
        }
        for i in 2..4 {
            params.reproduction_rates[i] *= 0.5;
            params.fertility[i] *= 0.5;
            params.competition_weights[i] *= 0.5;
        }

        for deme in 0..3 {
            let local = params.single_deme(deme);
            assert_eq!(local.n_demes, 1, "deme {deme}");
            let from_local = crate::config::SimConfig::assemble_deme(&bp, &local, &genetics, 0)
                .expect("local assembly is valid");
            let from_session =
                crate::config::SimConfig::assemble_deme(&bp, &params, &genetics, deme)
                    .expect("session assembly is valid");
            // Key scalars (including the derived equilibrium metrics, which
            // re-read every ecology input) and the vector segments.
            assert_eq!(from_local.carrying_capacity, from_session.carrying_capacity);
            assert_eq!(from_local.eggs_per_female, from_session.eggs_per_female);
            assert_eq!(from_local.sex_ratio, from_session.sex_ratio);
            assert_eq!(
                from_local.sperm_displacement_rate,
                from_session.sperm_displacement_rate
            );
            assert_eq!(
                from_local.low_density_growth_rate,
                from_session.low_density_growth_rate
            );
            assert_eq!(
                from_local.juvenile_growth_mode,
                from_session.juvenile_growth_mode
            );
            assert_eq!(
                from_local.expected_competition_strength,
                from_session.expected_competition_strength,
                "deme {deme}: equilibrium metrics re-read the local column"
            );
            assert_eq!(
                from_local.expected_survival_rate, from_session.expected_survival_rate,
                "deme {deme}"
            );
            assert_eq!(
                from_local.age_based_survival_rates,
                from_session.age_based_survival_rates
            );
            assert_eq!(
                from_local.age_based_mating_rates,
                from_session.age_based_mating_rates
            );
            assert_eq!(
                from_local.age_based_reproduction_rates,
                from_session.age_based_reproduction_rates
            );
            assert_eq!(
                from_local.female_age_based_fertility,
                from_session.female_age_based_fertility
            );
            assert_eq!(
                from_local.age_based_relative_competition_strength,
                from_session.age_based_relative_competition_strength
            );
        }

        // Sentinel columns survive the cut: derive-mode equilibrium and an
        // undeclared migration column stay empty; a declared migration
        // column yields exactly the deme's (2, A) segment.
        assert!(params.single_deme(1).equilibrium_distribution.is_empty());
        assert!(params.single_deme(1).migration_rate.is_empty());
        params.migration_rate = Params::tile_vec(&[0.01, 0.02, 0.01, 0.02], 3);
        assert_eq!(
            params.single_deme(2).migration_rate,
            vec![0.01, 0.02, 0.01, 0.02]
        );
    }
}
