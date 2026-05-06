use approx::assert_relative_eq;
use ndarray::{array, Array1};
use sprs::CsMat;

use spatial_xcorr::{gaussian_kernel, row_normalize, xcorr};

fn dense_from_csr(m: &CsMat<f32>) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; m.cols()]; m.rows()];
    for (i, row) in m.outer_iterator().enumerate() {
        for (j, &v) in row.iter() {
            out[i][j] = v;
        }
    }
    out
}

#[test]
fn kernel_two_cells_known_distance() {
    // (0,0) and (3,4) → d = 5; h = 5 → exp(-25/(2*25)) = exp(-0.5).
    let coords = array![[0.0_f32, 0.0], [3.0, 4.0]];
    let w = gaussian_kernel(coords.view(), 5.0, 10.0);
    let dense = dense_from_csr(&w);

    let expected = (-0.5_f32).exp();
    assert_relative_eq!(dense[0][1], expected, epsilon = 1e-5);
    assert_relative_eq!(dense[1][0], expected, epsilon = 1e-5);
    assert_eq!(dense[0][0], 0.0);
    assert_eq!(dense[1][1], 0.0);
}

#[test]
fn kernel_respects_r_max() {
    let coords = array![[0.0_f32, 0.0], [1.0, 0.0], [10.0, 0.0]];
    let w = gaussian_kernel(coords.view(), 1.0, 2.0);
    let dense = dense_from_csr(&w);

    assert!(dense[0][1] > 0.0);
    assert!(dense[1][0] > 0.0);
    assert_eq!(dense[0][2], 0.0);
    assert_eq!(dense[2][0], 0.0);
    assert_eq!(dense[1][2], 0.0);
    assert_eq!(dense[2][1], 0.0);
}

#[test]
fn kernel_3d_coords() {
    // (0,0,0) and (1,2,2) → d = sqrt(1+4+4) = 3; h = 3 → exp(-9/18) = exp(-0.5).
    let coords = array![[0.0_f32, 0.0, 0.0], [1.0, 2.0, 2.0]];
    let w = gaussian_kernel(coords.view(), 3.0, 5.0);
    let dense = dense_from_csr(&w);

    let expected = (-0.5_f32).exp();
    assert_relative_eq!(dense[0][1], expected, epsilon = 1e-5);
    assert_relative_eq!(dense[1][0], expected, epsilon = 1e-5);
}

#[test]
fn kernel_no_self_loops() {
    let coords = array![
        [0.0_f32, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let w = gaussian_kernel(coords.view(), 1.0, 5.0);
    let dense = dense_from_csr(&w);
    for (i, row) in dense.iter().enumerate() {
        assert_eq!(row[i], 0.0, "diagonal entry [{i},{i}] must be zero");
    }
}

#[test]
fn kernel_symmetric() {
    let coords = array![
        [0.0_f32, 0.0],
        [1.0, 0.5],
        [2.3, 1.1],
        [0.8, 2.2],
        [3.5, 0.4],
    ];
    let w = gaussian_kernel(coords.view(), 2.0, 5.0);
    let dense = dense_from_csr(&w);
    for (i, row) in dense.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_relative_eq!(v, dense[j][i], epsilon = 1e-6);
        }
    }
}

#[test]
fn kernel_empty_row_when_isolated() {
    let coords = array![[0.0_f32, 0.0], [1.0, 0.0], [100.0, 0.0]];
    let w = gaussian_kernel(coords.view(), 1.0, 2.0);

    // Cell 2 is too far from anyone — empty row in CSR.
    let row2 = w.outer_view(2).expect("row exists");
    let row2_sum: f32 = row2.data().iter().sum();
    assert_eq!(row2_sum, 0.0);

    // After row-normalize the row is still empty (no NaN).
    let normed = row_normalize(&w);
    let normed_dense = dense_from_csr(&normed);
    for &v in &normed_dense[2] {
        assert_eq!(v, 0.0);
        assert!(!v.is_nan());
    }
}

#[test]
fn kernel_compose_with_xcorr() {
    // 5 cells in a line at unit spacing. With h=1, r_max=3, every pair within
    // distance 3 is connected; the (0,4) pair (distance 4) is excluded.
    let coords = array![
        [0.0_f32, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ];
    let w = gaussian_kernel(coords.view(), 1.0, 3.0);
    let w_norm = row_normalize(&w);

    let x_a: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0];
    let x_b: Array1<f32> = array![0.0, 0.0, 0.0, 1.0, 1.0];

    let score = xcorr(x_a.view(), x_b.view(), &w_norm);
    assert!(score.is_finite(), "expected finite score, got {score}");
    assert!(
        score < -0.5,
        "expected strongly negative score, got {score}"
    );
}
