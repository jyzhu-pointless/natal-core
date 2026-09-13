//! Ecology snapshot and restore helpers shared by every session type.
//!
//! A memory checkpoint saves the ecology section only; the genetics section
//! is deliberately excluded because a checkpoint is a save, not an
//! uninstallation of genetic mods.  Keeping the helpers here stops the
//! discrete and spatial sessions from depending on the age-structured
//! session module.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::generated::ecology_parameters::ECOLOGY_SCALARS;
use crate::model::blueprint::Blueprint;
use crate::model::custom_fields::{custom_slots_from_python, custom_slots_to_python};
use crate::model::ecology::{EcologyParams, ECOLOGY_VECTORS};

/// Copy the ecology section of *params* into a fresh Python dict.
///
/// The genetics section is deliberately excluded: a memory checkpoint is a
/// save, not an uninstallation of genetic mods.
pub(crate) fn ecology_snapshot<'py>(
    py: Python<'py>,
    params: &EcologyParams,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    // Scalars and vectors are walked in generated wire order, so the dict is a
    // complete and order-stable description of the ecology section.
    for name in ECOLOGY_SCALARS {
        dict.set_item(name, params.get_scalar(name)?)?;
    }
    for name in ECOLOGY_VECTORS {
        dict.set_item(name, params.get_tensor(py, name)?)?;
    }
    // User-defined slots belong to the same atomic ecology checkpoint; slot 0
    // is the single panmictic deme column.
    dict.set_item(
        "custom_slots",
        custom_slots_to_python(py, &params.custom_slots[0])?,
    )?;
    Ok(dict)
}

/// Write an ecology snapshot dict back into *params*.
///
/// ## Errors
/// Returns ``PyValueError`` when a vector has the wrong size; each
/// ``tensor_write`` validates before committing, so the previous contents
/// are preserved for the failing field.
pub(crate) fn restore_ecology(
    params: &mut EcologyParams,
    bp: &Blueprint,
    ecology: &Bound<'_, PyAny>,
) -> PyResult<()> {
    // Build the restored section on a clone: every name/size is validated
    // while filling the candidate, and the live params are replaced only after
    // all fields succeeded, so a malformed dict cannot half-overwrite them.
    let mut candidate = params.clone();
    // Scalars go through ``apply`` so unknown names surface as errors.
    for name in ECOLOGY_SCALARS {
        let value: f64 = ecology.get_item(name)?.extract()?;
        candidate.apply(HashMap::from([(name.to_string(), value)]))?;
    }
    // Vectors are copied out first, then written through ``tensor_write``,
    // which checks the length against the blueprint before committing.
    for name in ECOLOGY_VECTORS {
        let values: Vec<f64> = ecology
            .get_item(name)?
            .extract::<PyReadonlyArray1<'_, f64>>()?
            .as_slice()?
            .to_vec();
        candidate.tensor_write(bp, name, values)?;
    }
    // Rebuild the declared custom types from Python; slot 0 is the single deme.
    candidate.custom_slots[0] = custom_slots_from_python(&ecology.get_item("custom_slots")?)?;
    // Commit only after every field validated.
    *params = candidate;
    Ok(())
}
