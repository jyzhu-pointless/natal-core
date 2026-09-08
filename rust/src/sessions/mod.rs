//! PyO3 session types and the execution-state machine they share.
//!
//! `status.rs` owns the `ExecutionStatus` transitions; the other modules
//! own one session type per lifecycle model, plus the ecology snapshot
//! helpers shared across session implementations.

pub(crate) mod age_structured;
pub(crate) mod discrete_generation;
pub(crate) mod ecology_snapshot;
pub(crate) mod spatial;
pub(crate) mod status;
