use approx::assert_relative_eq;
use ndarray::{array, Array1};
use sprs::{CsMat, TriMat};

use spatial_xcorr::{pearson, row_normalize, xcorr};

/// Build a CSR matrix from a dense `Vec<Vec<f32>>`. Test ergonomics only.
fn csr_from_dense(rows: &[Vec<f32>]) -> CsMat<f32> {
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let mut tri = TriMat::new((n_rows, n_cols));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    tri.to_csr()
}

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
fn pearson_basic() {
    let a = array![1.0_f32, 2.0, 3.0];
    let b = array![1.0_f32, 2.0, 3.0];
    let c = array![3.0_f32, 2.0, 1.0];
    assert_relative_eq!(pearson(a.view(), b.view()), 1.0, epsilon = 1e-5);
    assert_relative_eq!(pearson(a.view(), c.view()), -1.0, epsilon = 1e-5);
}

#[test]
fn pearson_zero_variance() {
    let constant = array![1.0_f32, 1.0, 1.0];
    let varying = array![1.0_f32, 2.0, 3.0];
    assert!(pearson(constant.view(), varying.view()).is_nan());
}

#[test]
fn row_normalize_simple() {
    let w = csr_from_dense(&[
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
    ]);
    let normed = row_normalize(&w);
    let dense = dense_from_csr(&normed);

    let expected = [
        vec![0.0, 0.5, 0.5],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.5, 0.5],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(dense[i][j], expected[i][j], epsilon = 1e-5);
        }
    }
}

#[test]
fn row_normalize_zero_row() {
    let w = csr_from_dense(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
    let normed = row_normalize(&w);
    let dense = dense_from_csr(&normed);

    assert_eq!(dense[0], vec![0.0, 0.0]);
    assert!(dense[0].iter().all(|v| !v.is_nan()));
    assert_relative_eq!(dense[1][0], 0.5, epsilon = 1e-5);
    assert_relative_eq!(dense[1][1], 0.5, epsilon = 1e-5);
}

/// Hand-validated reference: 5-cell path graph from the design doc.
///
/// Adjacency is the path 0—1—2—3—4. Row-normalizing gives row sums [1,2,2,2,1]
/// and a tridiagonal stochastic matrix. With `x_a = [1,1,0,0,0]` (left half
/// expressing) and `x_b = [0,0,0,1,1]` (right half), `W_norm @ x_b` smears the
/// right-half signal one step inward, producing `[0, 0, 0.5, 0.5, 1]`. The
/// Pearson correlation between `x_a` and that vector works out to
/// `-0.80 / sqrt(1.20 * 0.70) = -0.80 / sqrt(0.84) ≈ -0.8729`.
///
/// Reference value: `-0.873` (3 decimal places, matching design doc).
#[test]
fn xcorr_chain_reference() {
    let w = csr_from_dense(&[
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    ]);
    let w_norm = row_normalize(&w);

    let x_a: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0];
    let x_b: Array1<f32> = array![0.0, 0.0, 0.0, 1.0, 1.0];

    let score = xcorr(x_a.view(), x_b.view(), &w_norm);
    assert_relative_eq!(score, -0.873, epsilon = 1e-3);
}

/// Same path graph, but signal pair chosen to break the graph's reflection
/// symmetry. With `x_a = [1,1,0,0,0]` and `x_b = [0,0,0,1,1]` the two scores
/// would coincide because `x_b` is the mirror image of `x_a` and `W_norm` is
/// invariant under the same reflection. Using `x_b = [0,1,0,0,1]` (not a
/// mirror of `x_a`) makes the asymmetry observable.
#[test]
fn xcorr_asymmetric() {
    let w = csr_from_dense(&[
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    ]);
    let w_norm = row_normalize(&w);

    let x_a: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0];
    let x_b: Array1<f32> = array![0.0, 1.0, 0.0, 0.0, 1.0];

    let ab = xcorr(x_a.view(), x_b.view(), &w_norm);
    let ba = xcorr(x_b.view(), x_a.view(), &w_norm);
    assert!(
        (ab - ba).abs() > 1e-2,
        "expected asymmetric scores; got ab={ab}, ba={ba}",
    );
}

/// 4-cell fully-connected graph (no self-loops), `x = [1, 0, 0, 0]`.
///
/// Every row of W has sum 3, so W_norm has 1/3 in each off-diagonal entry.
/// Smoothed signal is `W_norm @ x = [0, 1/3, 1/3, 1/3]`. Both vectors have
/// mean 1/4. Centered: x is `[3/4, -1/4, -1/4, -1/4]`, smoothed is
/// `[-1/4, 1/12, 1/12, 1/12]`. Covariance `= -3/16 - 3/48 = -1/4`,
/// var_x = 3/4, var_sm = 1/12, denom = sqrt(1/16) = 1/4, so the score is
/// exactly `-1.0`. This catches off-by-one and self-loop errors.
#[test]
fn xcorr_self_with_dense_graph() {
    let w = csr_from_dense(&[
        vec![0.0, 1.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0, 0.0],
    ]);
    let w_norm = row_normalize(&w);

    let x: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
    let score = xcorr(x.view(), x.view(), &w_norm);
    assert_relative_eq!(score, -1.0, epsilon = 1e-5);
}
