//! Cached spatial neighbor lookup via kd-tree.
//!
//! `build_neighbor_list` returns one `Vec<(neighbor_idx, euclidean_distance)>`
//! per cell. The kd-tree is dropped before the function returns; only the
//! cached neighbor lists survive. TASK 005 will refactor `gaussian_kernel` to
//! consume this list rather than scan all pairs itself.

use kiddo::{KdTree, SquaredEuclidean};
use ndarray::ArrayView2;

#[derive(Debug, Clone)]
pub struct NeighborList {
    /// `neighbors[i]` lists the in-range neighbors of cell `i` as
    /// `(index, euclidean_distance)`. Self is always excluded.
    pub neighbors: Vec<Vec<(u32, f32)>>,
}

/// Build per-cell neighbor lists within `r_max`.
///
/// Coordinates may be 2D or 3D (column count is inferred). Distances are
/// returned as Euclidean — the kd-tree internally works in squared distance,
/// but we sqrt at insertion so callers don't have to. `r_max <= 0` panics;
/// non-2-or-3-dimensional coords also panic.
pub fn build_neighbor_list(coords: ArrayView2<f32>, r_max: f32) -> NeighborList {
    assert!(r_max > 0.0, "r_max must be positive, got {r_max}");
    match coords.ncols() {
        2 => build::<2>(coords, r_max),
        3 => build::<3>(coords, r_max),
        d => panic!("coords must be 2D or 3D, got {d}"),
    }
}

fn build<const K: usize>(coords: ArrayView2<f32>, r_max: f32) -> NeighborList {
    let n = coords.nrows();
    let mut points: Vec<[f32; K]> = Vec::with_capacity(n);
    for i in 0..n {
        let mut p = [0.0f32; K];
        for (k, slot) in p.iter_mut().enumerate() {
            *slot = coords[[i, k]];
        }
        points.push(p);
    }

    let mut tree: KdTree<f32, K> = KdTree::new();
    for (i, p) in points.iter().enumerate() {
        tree.add(p, i as u64);
    }

    let r_max_sq = r_max * r_max;
    let mut neighbors: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        let hits = tree.within::<SquaredEuclidean>(p, r_max_sq);
        let mut row: Vec<(u32, f32)> = Vec::with_capacity(hits.len().saturating_sub(1));
        for hit in hits {
            let idx = hit.item as usize;
            if idx == i {
                continue;
            }
            row.push((idx as u32, hit.distance.sqrt()));
        }
        neighbors.push(row);
    }
    NeighborList { neighbors }
}
