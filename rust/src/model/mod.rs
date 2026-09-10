//! Boundary data model shared by the Python contract and the kernels.
//!
//! [`blueprint::Blueprint`] is the frozen model specification (rebuild to
//! change), [`ecology::EcologyParams`] holds every runtime-mutable *ecology*
//! value as per-deme columns, and [`genetics::GeneticsTensors`] holds the
//! genotype-indexed *genetics* section that demes may share.
//! [`custom_fields::CustomSlot`] carries session-owned custom values, and
//! [`python`] / [`validation`] hold the extraction and domain checks all of
//! them use.

pub(crate) mod blueprint;
pub(crate) mod custom_fields;
pub(crate) mod ecology;
pub(crate) mod genetics;
pub(crate) mod python;
pub(crate) mod validation;
