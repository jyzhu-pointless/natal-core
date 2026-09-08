//! Declarative CSR hook interpretation and its transactional state.
//!
//! `interpreter.rs` evaluates the flat hook program inside the kernels;
//! `transaction.rs` owns the rollback/commit surface exposed to Python.

pub(crate) mod interpreter;
pub(crate) mod transaction;
