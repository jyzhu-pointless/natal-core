//! The genotype-indexed genetics tables.

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::model::blueprint::Blueprint;
use crate::model::python::extract_f64_vec;
use crate::model::validation::validate_tensor_values;

/// Contract names of the eight genetics tables carried by [`GeneticsTensors`].
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
/// tensor.  Genetics tensors live in [`GeneticsTensors`] instead of [`EcologyParams`].
#[must_use]
pub fn is_genetics_tensor(name: &str) -> bool {
    GENETICS_TENSORS.contains(&name)
}

/// The genetics section: eight genotype-indexed tables (flat row-major).
///
/// Demes with identical genetics share one ``GeneticsTensors`` through the
/// spatial session's variant bank, so heterogeneous models pay for the
/// tables once per *genetics variant*, not once per deme.
#[derive(Clone, Default, PartialEq)]
pub struct GeneticsTensors {
    pub viability_fitness: Vec<f64>,          // (2, A, Z)
    pub fecundity_fitness: Vec<f64>,          // (2, Z)
    pub sexual_selection_fitness: Vec<f64>,   // (Z, Z)
    pub zygote_viability_fitness: Vec<f64>,   // (2, Z)
    pub offspring_tensor: Vec<f64>,           // (Z, Z, Z)
    pub meiosis_map: Vec<f64>,                // (2, Z, G)
    pub female_ztype_compatibility: Vec<f64>, // (Z,)
    pub male_ztype_compatibility: Vec<f64>,   // (Z,)
}

impl GeneticsTensors {
    /// Extract the eight genetics tables from the Python contract object.
    ///
    /// ## Parameters
    /// - `obj`: A ``natal.contracts.Params`` instance.
    ///
    /// ## Returns
    /// An owned [`GeneticsTensors`] mirroring the contract tables.
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
        for (field, values) in &pending {
            validate_tensor_values(field, values)?;
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
        validate_tensor_values(name, &values)?;
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
