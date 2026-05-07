use approx::assert_relative_eq;
use ndarray::array;

use spatial_xcorr::build_neighbor_list;

fn line_coords_2d() -> ndarray::Array2<f32> {
    array![
        [0.0_f32, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ]
}

#[test]
fn path_graph_neighbors() {
    let coords = line_coords_2d();
    let nl = build_neighbor_list(coords.view(), 1.5);

    assert_eq!(nl.neighbors[0].len(), 1);
    assert_eq!(nl.neighbors[0][0].0, 1);
    assert_relative_eq!(nl.neighbors[0][0].1, 1.0, epsilon = 1e-5);

    assert_eq!(nl.neighbors[2].len(), 2);
    let mut row2 = nl.neighbors[2].clone();
    row2.sort_by_key(|(idx, _)| *idx);
    assert_eq!(row2[0].0, 1);
    assert_eq!(row2[1].0, 3);
    assert_relative_eq!(row2[0].1, 1.0, epsilon = 1e-5);
    assert_relative_eq!(row2[1].1, 1.0, epsilon = 1e-5);

    assert_eq!(nl.neighbors[4].len(), 1);
    assert_eq!(nl.neighbors[4][0].0, 3);
    assert_relative_eq!(nl.neighbors[4][0].1, 1.0, epsilon = 1e-5);
}

#[test]
fn r_max_excludes_far() {
    let coords = line_coords_2d();
    let nl = build_neighbor_list(coords.view(), 0.5);
    for (i, row) in nl.neighbors.iter().enumerate() {
        assert!(
            row.is_empty(),
            "cell {i} should have no neighbors, got {row:?}"
        );
    }
}

#[test]
fn r_max_includes_all() {
    let coords = line_coords_2d();
    let nl = build_neighbor_list(coords.view(), 100.0);
    for (i, row) in nl.neighbors.iter().enumerate() {
        assert_eq!(row.len(), 4, "cell {i} should see 4 neighbors");
    }
}

#[test]
fn no_self_in_neighbors() {
    let coords = line_coords_2d();
    let nl = build_neighbor_list(coords.view(), 100.0);
    for (i, row) in nl.neighbors.iter().enumerate() {
        for &(idx, _) in row {
            assert_ne!(idx as usize, i, "cell {i} appears in its own neighbor list");
        }
    }
}

#[test]
fn distances_are_euclidean_not_squared() {
    let coords = array![[0.0_f32, 0.0], [3.0, 4.0]];
    let nl = build_neighbor_list(coords.view(), 10.0);
    assert_eq!(nl.neighbors[0].len(), 1);
    assert_relative_eq!(nl.neighbors[0][0].1, 5.0, epsilon = 1e-5);
}

#[test]
fn works_in_3d() {
    let coords = array![[0.0_f32, 0.0, 0.0], [1.0, 2.0, 2.0]];
    let nl = build_neighbor_list(coords.view(), 5.0);
    assert_eq!(nl.neighbors[0].len(), 1);
    assert_relative_eq!(nl.neighbors[0][0].1, 3.0, epsilon = 1e-5);
}
