//! Per-model simulation kernels.
//!
//! Each lifecycle model owns one kernel module; `config.rs` holds the
//! assembled configuration shared by the age-structured kernels, and the
//! remaining modules hold the regulation, equilibrium, offspring and RNG
//! primitives those kernels call.

pub(crate) mod age_structured;
pub(crate) mod config;
// The curve library is a published contract (the built-in curve families
// plus their shared property checker), so it keeps the public reachability
// it had before the module was renamed to `density_regulation`.
pub mod density_regulation;
pub(crate) mod discrete_generation;
pub(crate) mod equilibrium;
pub(crate) mod offspring;
pub(crate) mod rng;
pub(crate) mod spatial;
pub(crate) mod state_reduce;
