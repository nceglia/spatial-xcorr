//! Library surface for `spatial-xcorr`: distance-resolved spatial gene-pair scoring.
//!
//! This module provides the asymmetric scoring primitive
//! `xcorr(a, b) = pearson(x_a, W_norm @ x_b)` along with the helpers it
//! composes: [`pearson`] and [`row_normalize`]. It also offers the building
//! blocks that turn a single score into a decay curve: [`gaussian_kernel`]
//! constructs a sparse coordinate adjacency at a chosen bandwidth, and
//! [`fit_exponential_decay`] fits `score(h) = A · exp(-h / λ)` to a sweep of
//! `(bandwidth, xcorr)` points via closed-form log-linear least squares.
//!
//! The graph multiplication in `xcorr` is implemented by manually iterating
//! sparse rows so this crate does not depend on `sprs`'s ndarray integration
//! (whose ndarray version may lag behind).

use ndarray::{Array1, ArrayView1, ArrayView2};
use sprs::{CsMat, TriMat};

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

/// Build an unnormalized Gaussian-kernel sparse adjacency from cell coordinates.
///
/// For every pair `(i, j)` with `i != j` and `||coords[i] - coords[j]|| <= r_max`,
/// sets `w_ij = exp(-d² / (2 · bandwidth²))`. The matrix is symmetric and has
/// no diagonal entries — a cell is not its own neighbor. Rows with no
/// in-range partners come back empty (zero row), which `row_normalize`
/// preserves as a zero row.
///
/// This implementation is brute-force `O(N²)` over all pairs. A kd-tree
/// neighbor lookup will replace it in a later task; for now this is fine for
/// the test sizes (≤200 cells).
///
/// Panics if `bandwidth <= 0`, `r_max < 0`, or `coords.nrows() == 0`.
pub fn gaussian_kernel(coords: ArrayView2<f32>, bandwidth: f32, r_max: f32) -> CsMat<f32> {
    assert!(
        bandwidth > 0.0,
        "bandwidth must be positive, got {bandwidth}"
    );
    assert!(r_max >= 0.0, "r_max must be non-negative, got {r_max}");
    let n = coords.nrows();
    assert!(n > 0, "coords must contain at least one cell");
    let n_dims = coords.ncols();

    let r_max_sq = r_max * r_max;
    let inv_2h2 = 1.0 / (2.0 * bandwidth * bandwidth);

    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0f32;
            for k in 0..n_dims {
                let diff = coords[[i, k]] - coords[[j, k]];
                d2 += diff * diff;
            }
            if d2 <= r_max_sq {
                let w = (-d2 * inv_2h2).exp();
                tri.add_triplet(i, j, w);
                tri.add_triplet(j, i, w);
            }
        }
    }
    tri.to_csr()
}

/// Result of fitting `score(h) = A · exp(-h / λ)` to a bandwidth sweep.
///
/// `decay_length` and `r_squared` are `NaN` when the fit is degenerate
/// (fewer than 2 finite-nonzero scores, no variation in bandwidth, flat or
/// growing curve). `sign` is the sign of the score at the smallest
/// bandwidth among the surviving points; `0` when the fit is degenerate or
/// the surviving score happens to be exactly zero.
#[derive(Debug, Clone, Copy)]
pub struct DecayFit {
    /// `A` in `score(h) = A · exp(-h / λ)`. Always non-negative; the sign is
    /// reported separately.
    pub amplitude: f32,
    /// `λ` in `score(h) = A · exp(-h / λ)`. `NaN` if the fit is degenerate
    /// or the curve is flat/growing.
    pub decay_length: f32,
    /// Coefficient of determination of the log-linear fit, clamped to `[0, 1]`.
    /// `NaN` if the fit is degenerate or all `log|score|` values are equal.
    pub r_squared: f32,
    /// `+1`, `-1`, or `0`.
    pub sign: i8,
}

const DEGENERATE_FIT: DecayFit = DecayFit {
    amplitude: f32::NAN,
    decay_length: f32::NAN,
    r_squared: f32::NAN,
    sign: 0,
};

/// Fit `score(h) = A · exp(-h / λ)` via weighted log-linear regression.
///
/// Filters to indices where both the bandwidth and score are finite and
/// `|score| > 0`, then regresses `log|score|` on `bandwidth` with simple
/// (unweighted) least squares. Sign is tracked separately so anti-coupled
/// gene pairs (negative scores) still produce a meaningful decay length.
///
/// Returns all-NaN with `sign = 0` when fewer than two points survive
/// filtering or when the bandwidths have no variance. Returns
/// `decay_length = NaN` (but a finite `r_squared`) when the fitted slope is
/// non-negative — i.e., the curve is flat or growing — since neither admits
/// a meaningful decay length.
///
/// Length mismatch between `bandwidths` and `scores` is a `debug_assert`.
pub fn fit_exponential_decay(bandwidths: &[f32], scores: &[f32]) -> DecayFit {
    debug_assert_eq!(bandwidths.len(), scores.len());

    // (bandwidth, score, ln|score|) for finite, nonzero-magnitude points.
    let pts: Vec<(f32, f32, f32)> = bandwidths
        .iter()
        .zip(scores.iter())
        .filter_map(|(&h, &s)| {
            if h.is_finite() && s.is_finite() && s.abs() > 0.0 {
                Some((h, s, s.abs().ln()))
            } else {
                None
            }
        })
        .collect();

    if pts.len() < 2 {
        return DEGENERATE_FIT;
    }

    // Sign comes from the score at the smallest *bandwidth* (not smallest index).
    let s_at_min_h = pts
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty after filter")
        .1;
    let sign: i8 = if s_at_min_h > 0.0 {
        1
    } else if s_at_min_h < 0.0 {
        -1
    } else {
        0
    };

    let n = pts.len() as f32;
    let mean_x: f32 = pts.iter().map(|&(h, _, _)| h).sum::<f32>() / n;
    let mean_y: f32 = pts.iter().map(|&(_, _, y)| y).sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut ss_tot = 0.0f32;
    for &(h, _, y) in &pts {
        let dx = h - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        ss_tot += dy * dy;
    }

    if var_x == 0.0 {
        // All bandwidths identical — the regression is undetermined.
        return DecayFit {
            amplitude: f32::NAN,
            decay_length: f32::NAN,
            r_squared: f32::NAN,
            sign,
        };
    }

    let slope = cov / var_x;
    let intercept = mean_y - slope * mean_x;
    let amplitude = intercept.exp();
    let decay_length = if slope < 0.0 { -1.0 / slope } else { f32::NAN };

    let r_squared = if ss_tot == 0.0 {
        f32::NAN
    } else {
        let mut ss_res = 0.0f32;
        for &(h, _, y) in &pts {
            let pred = intercept + slope * h;
            let resid = y - pred;
            ss_res += resid * resid;
        }
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    DecayFit {
        amplitude,
        decay_length,
        r_squared,
        sign,
    }
}
