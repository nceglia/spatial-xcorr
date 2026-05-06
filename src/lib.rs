//! Library surface for `spatial-xcorr`: distance-resolved spatial gene-pair scoring.
//!
//! This module provides the asymmetric scoring primitive
//! `xcorr(a, b) = pearson(x_a, W_norm @ x_b)` along with the two helpers it
//! composes: [`pearson`] and [`row_normalize`]. The graph multiplication is
//! implemented by manually iterating sparse rows so this crate does not depend
//! on `sprs`'s ndarray integration (whose ndarray version may lag behind).

use ndarray::{Array1, ArrayView1};
use sprs::CsMat;

/// Pearson correlation between two vectors of equal length.
///
/// Returns `f32::NAN` if either input has zero variance. Length mismatch is a
/// `debug_assert` only — callers are expected to validate shapes upstream.
pub fn pearson(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    debug_assert_eq!(u.len(), v.len());
    let n = u.len() as f32;
    let mean_u = u.sum() / n;
    let mean_v = v.sum() / n;

    let mut cov = 0.0f32;
    let mut var_u = 0.0f32;
    let mut var_v = 0.0f32;
    for (&ui, &vi) in u.iter().zip(v.iter()) {
        let du = ui - mean_u;
        let dv = vi - mean_v;
        cov += du * dv;
        var_u += du * du;
        var_v += dv * dv;
    }

    if var_u == 0.0 || var_v == 0.0 {
        f32::NAN
    } else {
        cov / (var_u * var_v).sqrt()
    }
}

/// Row-normalize a sparse adjacency matrix so each row sums to 1.
///
/// Rows that already sum to zero are left as zero rows — we propagate the empty
/// neighborhood downstream rather than producing NaN here.
pub fn row_normalize(w: &CsMat<f32>) -> CsMat<f32> {
    assert!(w.is_csr(), "row_normalize requires CSR storage");
    let (rows, cols) = (w.rows(), w.cols());

    let mut indptr: Vec<usize> = Vec::with_capacity(rows + 1);
    let mut indices: Vec<usize> = Vec::with_capacity(w.nnz());
    let mut data: Vec<f32> = Vec::with_capacity(w.nnz());
    indptr.push(0);

    for row in w.outer_iterator() {
        let row_sum: f32 = row.data().iter().sum();
        if row_sum > 0.0 {
            for (j, &v) in row.iter() {
                indices.push(j);
                data.push(v / row_sum);
            }
        }
        indptr.push(indices.len());
    }

    CsMat::new((rows, cols), indptr, indices, data)
}

/// Asymmetric spatial cross-correlation: `pearson(x_a, W_norm @ x_b)`.
///
/// `w_norm` must already be row-normalized — see [`row_normalize`]. Length of
/// each expression vector must match the number of rows in `w_norm`.
pub fn xcorr(x_a: ArrayView1<f32>, x_b: ArrayView1<f32>, w_norm: &CsMat<f32>) -> f32 {
    debug_assert_eq!(x_a.len(), w_norm.rows());
    debug_assert_eq!(x_b.len(), w_norm.rows());
    debug_assert!(w_norm.is_csr());

    let n = w_norm.rows();
    let mut smoothed = Array1::<f32>::zeros(n);
    for (i, row) in w_norm.outer_iterator().enumerate() {
        let mut acc = 0.0f32;
        for (j, &v) in row.iter() {
            acc += v * x_b[j];
        }
        smoothed[i] = acc;
    }

    pearson(x_a, smoothed.view())
}
