//! History storage, parameter logs, and the shared numerical projection.
//!
//! `history.rs` owns the ring, its row data, and the shared history handle;
//! `parameter_log.rs` owns the parameter log and log value conversion;
//! `observation.rs` owns the numerical projection and its Python entry
//! point.

pub(crate) mod history;
pub(crate) mod observation;
pub(crate) mod parameter_log;
