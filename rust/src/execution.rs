//! Shared execution states for all native session shapes.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// A session can resume only from a valid Ready boundary.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum Execution {
    #[default]
    Ready,
    Running,
    Stopped,
    Failed,
}

impl Execution {
    /// Check the transition before any model state can be mutated.
    pub fn begin(&mut self) -> PyResult<()> {
        if *self != Self::Ready {
            return Err(PyRuntimeError::new_err(
                "Session is not Ready; restore a checkpoint or reset before run",
            ));
        }
        *self = Self::Running;
        Ok(())
    }

    /// Expose a stable status without lending any mutable handle.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::Running => "Running",
            Self::Stopped => "Stopped",
            Self::Failed => "Failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Execution;

    #[test]
    fn begin_requires_ready_and_preserves_rejected_state() {
        let mut ready = Execution::Ready;
        assert!(ready.begin().is_ok());
        assert_eq!(ready.name(), "Running");
        for (mut state, name) in [
            (Execution::Running, "Running"),
            (Execution::Stopped, "Stopped"),
            (Execution::Failed, "Failed"),
        ] {
            assert!(state.begin().is_err());
            assert_eq!(state.name(), name);
        }
    }
}
