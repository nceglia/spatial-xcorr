use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use spatial_xcorr::{read_dataset, IoError};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn tmpdir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("sxc-{tag}-{pid}-{nanos}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn reads_tiny_fixture() {
    let dir = fixture("tiny");
    let ds = read_dataset(
        &dir.join("counts.mtx"),
        &dir.join("genes.tsv"),
        &dir.join("cells.tsv"),
        &dir.join("coords.csv"),
    )
    .expect("read tiny fixture");

    assert_eq!((ds.counts.rows(), ds.counts.cols()), (5, 3));
    assert_eq!(ds.genes.len(), 3);
    assert_eq!(ds.genes[0], "GeneA");
    assert_eq!(ds.cells[2], "cell2");
    assert_eq!(ds.coords.shape(), &[5, 2]);
    assert_eq!(ds.coords[[3, 0]], 3.0);
}

#[test]
fn reads_3d_coords() {
    let dir = fixture("tiny_3d");
    let ds = read_dataset(
        &dir.join("counts.mtx"),
        &dir.join("genes.tsv"),
        &dir.join("cells.tsv"),
        &dir.join("coords.csv"),
    )
    .expect("read tiny_3d fixture");
    assert_eq!(ds.coords.shape(), &[5, 3]);
    assert_eq!(ds.coords[[2, 2]], 0.5);
}

#[test]
fn rejects_dim_mismatch() {
    let src = fixture("tiny");
    let dst = tmpdir("dim-mismatch");
    fs::copy(src.join("counts.mtx"), dst.join("counts.mtx")).unwrap();
    fs::copy(src.join("genes.tsv"), dst.join("genes.tsv")).unwrap();
    fs::copy(src.join("coords.csv"), dst.join("coords.csv")).unwrap();
    // Wrong cell count: 4 instead of 5.
    fs::write(dst.join("cells.tsv"), "cell0\ncell1\ncell2\ncell3\n").unwrap();

    let result = read_dataset(
        &dst.join("counts.mtx"),
        &dst.join("genes.tsv"),
        &dst.join("cells.tsv"),
        &dst.join("coords.csv"),
    );
    let _ = fs::remove_dir_all(&dst);

    match result {
        Err(IoError::DimMismatch { expected, got }) => {
            assert_eq!(expected, (4, 3));
            assert_eq!(got, (5, 3));
        }
        other => panic!("expected DimMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_4d_coords() {
    let src = fixture("tiny");
    let dst = tmpdir("4d-coords");
    fs::copy(src.join("counts.mtx"), dst.join("counts.mtx")).unwrap();
    fs::copy(src.join("genes.tsv"), dst.join("genes.tsv")).unwrap();
    fs::copy(src.join("cells.tsv"), dst.join("cells.tsv")).unwrap();
    fs::write(
        dst.join("coords.csv"),
        "0,0,0,0\n1,0,0,0\n2,0,0,0\n3,0,0,0\n4,0,0,0\n",
    )
    .unwrap();

    let result = read_dataset(
        &dst.join("counts.mtx"),
        &dst.join("genes.tsv"),
        &dst.join("cells.tsv"),
        &dst.join("coords.csv"),
    );
    let _ = fs::remove_dir_all(&dst);

    match result {
        Err(IoError::Coords(msg)) => {
            assert!(
                msg.contains('4'),
                "expected message to mention 4 columns, got: {msg}"
            );
        }
        other => panic!("expected IoError::Coords, got {other:?}"),
    }
}

#[test]
fn mtx_values_correct() {
    let dir = fixture("tiny");
    let ds = read_dataset(
        &dir.join("counts.mtx"),
        &dir.join("genes.tsv"),
        &dir.join("cells.tsv"),
        &dir.join("coords.csv"),
    )
    .expect("read tiny fixture");

    let dense = ds.counts.to_dense();
    assert_eq!(dense[[0, 0]], 1.0); // cell0/GeneA
    assert_eq!(dense[[0, 2]], 2.0); // cell0/GeneC
    assert_eq!(dense[[2, 0]], 0.0); // cell2/GeneA — empty entry
    assert_eq!(dense[[4, 1]], 1.0); // cell4/GeneB
    assert_eq!(dense[[4, 2]], 2.0); // cell4/GeneC
}
