use super::ExecutionStatus;

#[test]
fn begin_requires_ready_and_preserves_rejected_state() {
    let mut ready = ExecutionStatus::Ready;
    assert!(ready.begin().is_ok());
    assert_eq!(ready.name(), "Running");
    for (mut state, name) in [
        (ExecutionStatus::Running, "Running"),
        (ExecutionStatus::Stopped, "Stopped"),
        (ExecutionStatus::Failed, "Failed"),
    ] {
        assert!(state.begin().is_err());
        assert_eq!(state.name(), name);
    }
}
