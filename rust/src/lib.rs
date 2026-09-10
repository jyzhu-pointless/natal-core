//! Rust backend for NATAL Core.
//!
//! The crate exposes PyO3 sessions for age-structured, discrete-generation /
//! Wright-Fisher, and spatial multi-deme simulations.  Configuration is
//! snapshotted into plain Rust structs and CSR declarative hooks are
//! interpreted directly in the kernels.

mod generated;
mod hooks;
pub mod kernels;
mod model;
mod output;
mod python;
#[cfg(test)]
#[path = "../tests/unit/runtime_boundaries.rs"]
mod runtime_boundaries;
mod sessions;

use pyo3::prelude::*;

use crate::output::history::HistoryStore;
use crate::output::parameter_log::ParameterLog;
use crate::sessions::age_structured::AgeStructuredSession;
use crate::sessions::discrete_generation::DiscreteGenerationSession;
use crate::sessions::spatial::SpatialSession;

#[pymodule]
fn _engine_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::kernels::offspring::compute_offspring_tensor;

    module.add_class::<HistoryStore>()?;
    module.add_class::<ParameterLog>()?;
    module.add_function(wrap_pyfunction!(
        output::observation::project_observation,
        module
    )?)?;
    module.add_class::<AgeStructuredSession>()?;
    module.add_class::<DiscreteGenerationSession>()?;
    module.add_class::<SpatialSession>()?;
    module.add_function(wrap_pyfunction!(python::equilibrium_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(python::equilibrium_metrics_flat, module)?)?;
    module.add_function(wrap_pyfunction!(python::migrate_csr_deterministic, module)?)?;
    module.add_function(wrap_pyfunction!(python::migrate_csr_stochastic, module)?)?;
    module.add_function(wrap_pyfunction!(compute_offspring_tensor, module)?)?;
    Ok(())
}
