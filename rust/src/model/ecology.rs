//! Every runtime-mutable ecology value, stored as per-deme columns.
//!
//! Slice-5 stage-2 layout (structure-of-arrays across demes):
//!
//! - Every ecology scalar is a `Vec<f64>` column of length `n_demes`
//!   (a panmictic model is one deme, so its columns have length 1 and the
//!   kernels read entry 0 — the flat contents of a length-1 column are
//!   bit-identical to the pre-columnization scalars).
//! - Every ecology vector carries a leading deme dimension
//!   (`(n_demes, ...)` flattened row-major).
//!
//! Params mutate *in place* through `EcologyParams::apply` /
//! `EcologyParams::tensor_write` / `EcologyParams::pull_fields` — no
//! fingerprints, no session rebuilds, no RNG resets.
//!
//! Field names and shapes mirror the Python contract in
//! `natal/contracts/params.py` (CONTRACTS_VERSION >= 2).

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::generated::ecology_parameters::{ECOLOGY_SCALAR_COLUMNS, ECO_PARAM_COLUMNS};
use crate::model::blueprint::Blueprint;
use crate::model::custom_fields::{extract_custom_slots, CustomSlot};
use crate::model::genetics::{is_genetics_tensor, GeneticsTensors};
use crate::model::python::{extract_f64, extract_f64_vec, extract_i64};
use crate::model::validation::{validate_scalar_value, validate_tensor_values};

/// Whether a contract field name denotes an ecology tensor (vector) field
/// owned by [`EcologyParams`].  Genetics tensors are *not* ecology tensors.
fn is_ecology_tensor(name: &str) -> bool {
    matches!(name, |"survival_rates"| "mating_rates"
        | "reproduction_rates"
        | "fertility"
        | "competition_weights"
        | "equilibrium_distribution"
        | "migration_rate")
}

/// Every runtime-mutable *ecology* value, stored as per-deme columns.
///
/// Column layout (slice-5 stage 2): scalar columns have length ``n_demes``
/// and vector columns are ``(n_demes, ...)`` flattened row-major.  A
/// panmictic model owns one deme, so its columns have length 1 and every
/// kernel reads entry 0 — bit-identical to the pre-columnization layout.
///
/// The genetics section lives in [`GeneticsTensors`] (shared across demes via
/// the session's variant bank); all writes go through the validated
/// channels ([`EcologyParams::apply`], [`EcologyParams::tensor_write`],
/// [`EcologyParams::pull_fields`]).
/// Ecology vector field names carried by a memory checkpoint.
///
/// ``migration_rate`` is the spatial rate column folded into the params
/// contract (slice 5); restoring it keeps a checkpoint a complete save of
/// the ecology section.
pub(crate) const ECOLOGY_VECTORS: [&str; 7] = [
    "survival_rates",
    "mating_rates",
    "reproduction_rates",
    "fertility",
    "competition_weights",
    "equilibrium_distribution",
    "migration_rate",
];

#[derive(Clone)]
pub struct EcologyParams {
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
    pub equilibrium_distribution: Vec<f64>, // (n_demes, 2, A) or empty
    /// Per-deme declaration presence; a derived neighbor must not become zero.
    pub equilibrium_declared: Vec<bool>,
    pub migration_rate: Vec<f64>, // (n_demes, 2, A) or empty (not declared)
    // -- custom slots --
    pub custom_slots: Vec<HashMap<String, CustomSlot>>,
}

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

impl EcologyParams {
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

    /// Extract one deme's ecology into a fresh single-deme ``EcologyParams``.
    ///
    /// Spatial parallel ticks need a per-deme mutable write target for
    /// ``Op.set_param`` commits, but the session's ecology columns cannot
    /// be borrowed ``&mut`` by several demes at once.  Each deme therefore
    /// ticks against this private single-column copy; ``assemble_deme``
    /// reads position 0, whose contents are exactly the values the source
    /// column held at this deme (so a fresh assembly is numerically
    /// identical to the session's per-deme config).  Genetics are *not*
    /// carried — the caller shares the immutable [`GeneticsTensors`].
    ///
    /// ## Parameters
    /// - `deme`: Deme whose column entries and vector segments are copied.
    ///
    /// ## Returns
    /// A ``EcologyParams`` with ``n_demes == 1`` holding only that deme's ecology
    /// (scalar custom slots are shared by clone).
    ///
    /// ## Panics
    /// Panics when *deme* is out of range for the scalar columns.
    /// Merge a validated single-deme candidate without touching sibling columns.
    pub fn replace_deme(&mut self, deme: usize, candidate: &EcologyParams) {
        self.carrying_capacity[deme] = candidate.carrying_capacity[0];
        self.eggs_per_female[deme] = candidate.eggs_per_female[0];
        self.sex_ratio[deme] = candidate.sex_ratio[0];
        self.sperm_displacement_rate[deme] = candidate.sperm_displacement_rate[0];
        self.low_density_growth_rate[deme] = candidate.low_density_growth_rate[0];
        self.growth_mode[deme] = candidate.growth_mode[0];
        self.external_expected_eggs[deme] = candidate.external_expected_eggs[0];
        // Required columns were shape-validated before this candidate commit.
        let width = candidate.survival_rates.len();
        self.survival_rates[deme * width..(deme + 1) * width]
            .copy_from_slice(&candidate.survival_rates);
        // Required columns were shape-validated before this candidate commit.
        let width = candidate.mating_rates.len();
        self.mating_rates[deme * width..(deme + 1) * width]
            .copy_from_slice(&candidate.mating_rates);
        // Required columns were shape-validated before this candidate commit.
        let width = candidate.reproduction_rates.len();
        self.reproduction_rates[deme * width..(deme + 1) * width]
            .copy_from_slice(&candidate.reproduction_rates);
        // Required columns were shape-validated before this candidate commit.
        let width = candidate.fertility.len();
        self.fertility[deme * width..(deme + 1) * width].copy_from_slice(&candidate.fertility);
        // Required columns were shape-validated before this candidate commit.
        let width = candidate.competition_weights.len();
        self.competition_weights[deme * width..(deme + 1) * width]
            .copy_from_slice(&candidate.competition_weights);
        self.equilibrium_declared[deme] = candidate.equilibrium_declared[0];
        if candidate.equilibrium_declared[0] {
            let width = candidate.equilibrium_distribution.len();
            if self.equilibrium_distribution.is_empty() {
                self.equilibrium_distribution = vec![0.0; width * self.n_demes];
            }
            self.equilibrium_distribution[deme * width..(deme + 1) * width]
                .copy_from_slice(&candidate.equilibrium_distribution);
        }
        if !self.equilibrium_declared.iter().any(|declared| *declared) {
            self.equilibrium_distribution.clear();
        }
        if !candidate.migration_rate.is_empty() {
            let width = candidate.migration_rate.len();
            if self.migration_rate.is_empty() {
                self.migration_rate = vec![0.0; width * self.n_demes];
            }
            self.migration_rate[deme * width..(deme + 1) * width]
                .copy_from_slice(&candidate.migration_rate);
        }
        self.custom_slots[deme] = candidate.custom_slots[0].clone();
    }

    pub fn single_deme(&self, deme: usize) -> EcologyParams {
        EcologyParams {
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
            equilibrium_distribution: if self.equilibrium_declared[deme] {
                Self::cut_deme_segment(&self.equilibrium_distribution, self.n_demes, deme)
            } else {
                Vec::new()
            },
            equilibrium_declared: vec![self.equilibrium_declared[deme]],
            // The migration-rate column is cut like every other vector
            // column: a deme's lifecycle consumes only its own (2, A) rate
            // segment; the empty sentinel stays empty.
            migration_rate: Self::cut_deme_segment(&self.migration_rate, self.n_demes, deme),
            custom_slots: vec![self.custom_slots[deme].clone()],
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
    /// An owned [`EcologyParams`] with columnized ecology.
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
            equilibrium_declared: vec![!equilibrium.is_empty(); n_demes],
            equilibrium_distribution: if equilibrium.is_empty() {
                Vec::new()
            } else {
                Self::tile_vec(&equilibrium, n_demes)
            },
            migration_rate: extract_f64_vec(obj, "migration_rate")?,
            custom_slots: vec![extract_custom_slots(obj)?; n_demes],
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
            equilibrium_declared: vec![false; n_demes],
            migration_rate: Vec::new(),
            custom_slots: vec![HashMap::new(); n_demes],
        };
        let dict = obj.downcast::<PyDict>()?;
        let mut declared_override = None;
        let mut seen = std::collections::HashSet::new();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            if !seen.insert(key.clone()) {
                return Err(PyValueError::new_err(format!(
                    "duplicate ecology column {key:?}"
                )));
            }
            if key == "equilibrium_declared" {
                let values = value
                    .extract::<PyReadonlyArrayDyn<'_, i64>>()?
                    .as_slice()?
                    .to_vec();
                Self::expect_column_len(&key, values.len(), n_demes)?;
                if values.iter().any(|value| !matches!(value, 0 | 1)) {
                    return Err(PyValueError::new_err(
                        "equilibrium_declared must contain 0 or 1",
                    ));
                }
                declared_override = Some(values.into_iter().map(|value| value == 1).collect());
                continue;
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
                    "equilibrium_distribution" => {
                        params.equilibrium_declared.fill(!values.is_empty());
                        params.equilibrium_distribution = values;
                    }
                    "migration_rate" => params.migration_rate = values,
                    _ => unreachable!("name matched the tensor column list"),
                }
            } else {
                return Err(PyKeyError::new_err(format!(
                    "unknown ecology column {key:?}"
                )));
            }
        }
        if let Some(declared) = declared_override {
            params.equilibrium_declared = declared;
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
        if name == "equilibrium_distribution" {
            self.equilibrium_declared[deme] = true;
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
        genetics: Option<&mut GeneticsTensors>,
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
                let expected = GeneticsTensors::expected_len(bp, field)?;
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
        let pending_custom = if wants_custom_slots {
            Some(extract_custom_slots(source)?)
        } else {
            None
        };
        for (name, value) in &pending_scalars {
            validate_scalar_value(name, *value)?;
        }
        for (name, values) in pending_tensors.iter().chain(pending_genetics.iter()) {
            validate_tensor_values(name, values)?;
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
                if field == "equilibrium_distribution" {
                    self.equilibrium_declared[deme] = false;
                    if !self.equilibrium_declared.iter().any(|declared| *declared) {
                        self.equilibrium_distribution.clear();
                    }
                }
                continue;
            }
            self.write_deme_tensor(bp, field, deme, values)?;
        }
        if let Some(sink) = genetics {
            for (field, values) in &pending_genetics {
                sink.tensor_write(bp, field, values.clone())?;
            }
        }
        if let Some(slots) = pending_custom {
            self.custom_slots[deme] = slots;
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

impl EcologyParams {
    /// Validate that every column and vector field matches the
    /// blueprint-derived size.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on the first size mismatch.
    pub fn validate(&self, bp: &Blueprint) -> PyResult<()> {
        if self.equilibrium_declared.len() != self.n_demes
            || (self.equilibrium_distribution.is_empty()
                && self.equilibrium_declared.iter().any(|value| *value))
        {
            return Err(PyValueError::new_err(
                "equilibrium declaration presence does not match its columns",
            ));
        }
        if self.custom_slots.len() != self.n_demes {
            return Err(PyValueError::new_err(
                "custom-slot column must have one entry per deme",
            ));
        }

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
        for (name, value) in &writes {
            self.scalar_ref(name, 0)?;
            validate_scalar_value(name, *value)?;
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
    /// session routes them to its [`GeneticsTensors`]).
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown or genetics fields and
    /// ``PyValueError`` on size mismatch; the previous contents are
    /// preserved on failure.
    pub fn tensor_write(&mut self, bp: &Blueprint, name: &str, values: Vec<f64>) -> PyResult<()> {
        validate_tensor_values(name, &values)?;
        if is_genetics_tensor(name) {
            return Err(PyKeyError::new_err(format!(
                "{name:?} is a genetics tensor; route it to the session genetics bank"
            )));
        }
        let expected = self.expected_len(bp, name)?;
        // The derive-mode sentinel for equilibrium_distribution (empty) is
        // legal content: writing full length widens it, writing empty keeps
        // it, and any other size is rejected.
        let keeps_empty_sentinel = (name == "equilibrium_distribution" && values.is_empty())
            || (name == "migration_rate" && {
                let current_empty = match name {
                    "equilibrium_distribution" => self.equilibrium_distribution.is_empty(),
                    _ => self.migration_rate.is_empty(),
                };
                current_empty && values.is_empty()
            });
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
            "equilibrium_distribution" => {
                self.equilibrium_declared.fill(!values.is_empty());
                self.equilibrium_distribution = values;
            }
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

    /// Snapshot the canonical ECO param values for one deme column.
    ///
    /// Borrow-splitting twin of ``AgeStructuredSession::eco_values``: reads only
    /// the params, so callers holding a field-wise borrow of the session
    /// can still assemble the scratch row.
    pub(crate) fn eco_values_row(
        &self,
        deme: usize,
    ) -> [f64; crate::hooks::interpreter::N_ECO_PARAMS] {
        let mut values = [0.0; crate::hooks::interpreter::N_ECO_PARAMS];
        for (id, slot) in values.iter_mut().enumerate() {
            *slot = self.eco_value(id, deme);
        }
        values
    }

    /// Copy one ecology vector column (full column, all demes).
    fn ecology_vector_copy(&self, name: &str) -> PyResult<Vec<f64>> {
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
        Ok(v.clone())
    }

    /// Snapshot the full ecology section as parallel word vectors.
    ///
    /// Rust-native twin of the Python-dict ``ecology_snapshot``: native
    /// session checkpoints capture ecology without a Python round trip.
    /// Scalars follow the ``ECOLOGY_SCALAR_COLUMNS``
    /// wire order (deme 0); vectors follow ``ECOLOGY_VECTORS`` (full
    /// columns).
    pub(crate) fn ecology_snapshot_words(&self) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let mut scalars =
            Vec::with_capacity(crate::generated::ecology_parameters::ECOLOGY_SCALARS.len());
        for name in crate::generated::ecology_parameters::ECOLOGY_SCALARS {
            scalars.push(self.get_scalar(name)?);
        }
        let mut vectors = Vec::with_capacity(ECOLOGY_VECTORS.len());
        for name in ECOLOGY_VECTORS {
            vectors.push(self.ecology_vector_copy(name)?);
        }
        Ok((scalars, vectors))
    }

    /// Restore the ecology section from parallel word vectors.
    ///
    /// Companion of [`EcologyParams::ecology_snapshot_words`]: per-field
    /// validation through ``apply`` / ``tensor_write`` keeps prior contents
    /// on failure, exactly like the dict-based ``restore_ecology``.
    pub(crate) fn ecology_restore_words(
        &mut self,
        bp: &Blueprint,
        scalars: &[f64],
        vectors: &[Vec<f64>],
    ) -> PyResult<()> {
        for (name, value) in crate::generated::ecology_parameters::ECOLOGY_SCALARS
            .iter()
            .zip(scalars.iter())
        {
            self.apply(HashMap::from([(name.to_string(), *value)]))?;
        }
        for (name, values) in ECOLOGY_VECTORS.iter().zip(vectors.iter()) {
            self.tensor_write(bp, name, values.clone())?;
        }
        Ok(())
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
/// [`GeneticsTensors`], ecology names from [`EcologyParams`].
///
/// ## Errors
/// Returns ``PyKeyError`` for unknown names.
pub fn session_get_tensor<'py>(
    py: Python<'py>,
    params: &EcologyParams,
    genetics: &GeneticsTensors,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if is_genetics_tensor(name) {
        genetics.get_tensor(py, name)
    } else {
        params.get_tensor(py, name)
    }
}

/// Session-level tensor write routing: genetics names write into the
/// [`GeneticsTensors`], ecology names into [`EcologyParams`].
///
/// ## Errors
/// Returns ``PyKeyError`` for unknown names and ``PyValueError`` on size
/// mismatch; previous contents are preserved on failure.
pub fn session_tensor_write(
    bp: &Blueprint,
    params: &mut EcologyParams,
    genetics: &mut GeneticsTensors,
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
#[path = "../../tests/unit/model/ecology.rs"]
mod tests;
