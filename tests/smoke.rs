use std::process::Command;

#[test]
fn binary_runs() {
    let status = Command::new(env!("CARGO_BIN_EXE_spatial-xcorr"))
        .status()
        .expect("failed to spawn spatial-xcorr binary");
    assert!(status.success(), "binary exited with {status}");
}
