//! File I/O for the dataset bundle: counts (Matrix Market), genes/cells (line
//! files), and coordinates (headerless CSV).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array2;
use sprs::{CsMat, TriMat};

/// All four input files combined, with shapes already cross-validated.
#[derive(Debug)]
pub struct Dataset {
    pub counts: CsMat<f32>,
    pub genes: Vec<String>,
    pub cells: Vec<String>,
    pub coords: Array2<f32>,
}

/// Errors produced by [`read_dataset`] and its helpers.
#[derive(Debug)]
pub enum IoError {
    Mtx(String),
    LineFile(String),
    Coords(String),
    DimMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Mtx(m) => write!(f, "matrix market: {m}"),
            IoError::LineFile(m) => write!(f, "line file: {m}"),
            IoError::Coords(m) => write!(f, "coords: {m}"),
            IoError::DimMismatch { expected, got } => {
                let transpose_hint =
                    if got.0 == expected.1 && got.1 == expected.0 && expected.0 != expected.1 {
                        " (did you mean to transpose?)"
                    } else {
                        ""
                    };
                write!(
                    f,
                    "dimension mismatch: expected {expected:?}, got {got:?}{transpose_hint}",
                )
            }
        }
    }
}

impl std::error::Error for IoError {}

/// Read the four-file dataset bundle.
///
/// All shapes are validated relative to one another: `counts` must be
/// `(cells.len(), genes.len())`, and `coords.nrows()` must equal
/// `cells.len()`.
pub fn read_dataset(
    counts_path: &Path,
    genes_path: &Path,
    cells_path: &Path,
    coords_path: &Path,
) -> Result<Dataset, IoError> {
    let counts = read_mtx(counts_path)?;
    let genes = read_lines(genes_path)?;
    let cells = read_lines(cells_path)?;
    let coords = read_coords(coords_path)?;

    let counts_shape = (counts.rows(), counts.cols());
    let expected_counts = (cells.len(), genes.len());
    if counts_shape != expected_counts {
        return Err(IoError::DimMismatch {
            expected: expected_counts,
            got: counts_shape,
        });
    }
    if coords.nrows() != cells.len() {
        return Err(IoError::DimMismatch {
            expected: (cells.len(), coords.ncols()),
            got: (coords.nrows(), coords.ncols()),
        });
    }

    Ok(Dataset {
        counts,
        genes,
        cells,
        coords,
    })
}

/// Minimal Matrix Market coordinate-format reader.
///
/// Accepts `integer`, `real`, and `pattern` fields with `general` symmetry.
/// Casts everything to `f32`. Symmetric/Hermitian variants and `array` format
/// are deliberately rejected — the counts matrix is always coordinate/general.
fn read_mtx(path: &Path) -> Result<CsMat<f32>, IoError> {
    let file =
        File::open(path).map_err(|e| IoError::Mtx(format!("open {}: {e}", path.display())))?;
    let mut reader = BufReader::new(file);

    let mut header = String::new();
    reader
        .read_line(&mut header)
        .map_err(|e| IoError::Mtx(format!("read header: {e}")))?;
    if header.is_empty() {
        return Err(IoError::Mtx("empty file".into()));
    }
    let header = header.trim_end();
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 5 || parts[0] != "%%MatrixMarket" {
        return Err(IoError::Mtx(format!(
            "missing or malformed banner: {header}"
        )));
    }
    if parts[1] != "matrix" || parts[2] != "coordinate" {
        return Err(IoError::Mtx(format!(
            "only 'matrix coordinate' supported, got: {header}"
        )));
    }
    let field = parts[3];
    let symmetry = parts[4];
    if symmetry != "general" {
        return Err(IoError::Mtx(format!("symmetry '{symmetry}' not supported")));
    }
    let value_per_row: usize = match field {
        "pattern" => 0,
        "integer" | "real" => 1,
        other => {
            return Err(IoError::Mtx(format!("field '{other}' not supported")));
        }
    };

    // Skip comment + blank lines, find the size line.
    let mut size_line = String::new();
    loop {
        size_line.clear();
        let n = reader
            .read_line(&mut size_line)
            .map_err(|e| IoError::Mtx(format!("read: {e}")))?;
        if n == 0 {
            return Err(IoError::Mtx("missing size line".into()));
        }
        let trimmed = size_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        break;
    }
    let mut size_parts = size_line.split_whitespace();
    let m: usize = parse_field(&mut size_parts, &size_line, "rows")?;
    let n_cols: usize = parse_field(&mut size_parts, &size_line, "cols")?;
    let nnz: usize = parse_field(&mut size_parts, &size_line, "nnz")?;

    let mut tri = TriMat::new((m, n_cols));
    let mut count = 0usize;
    let mut buf = String::new();
    loop {
        buf.clear();
        let read = reader
            .read_line(&mut buf)
            .map_err(|e| IoError::Mtx(format!("read: {e}")))?;
        if read == 0 {
            break;
        }
        let trimmed = buf.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        let mut tok = trimmed.split_whitespace();
        let i: usize = parse_field(&mut tok, trimmed, "row index")?;
        let j: usize = parse_field(&mut tok, trimmed, "col index")?;
        let v: f32 = if value_per_row == 0 {
            1.0
        } else {
            tok.next()
                .ok_or_else(|| IoError::Mtx(format!("missing value: {trimmed}")))?
                .parse::<f32>()
                .map_err(|e| IoError::Mtx(format!("bad value '{trimmed}': {e}")))?
        };
        if i == 0 || j == 0 || i > m || j > n_cols {
            return Err(IoError::Mtx(format!(
                "index out of range in line: {trimmed}"
            )));
        }
        tri.add_triplet(i - 1, j - 1, v);
        count += 1;
    }
    if count != nnz {
        return Err(IoError::Mtx(format!("expected {nnz} entries, got {count}")));
    }

    Ok(tri.to_csr())
}

fn parse_field<'a, T: std::str::FromStr>(
    iter: &mut impl Iterator<Item = &'a str>,
    line: &str,
    what: &str,
) -> Result<T, IoError>
where
    T::Err: std::fmt::Display,
{
    iter.next()
        .ok_or_else(|| IoError::Mtx(format!("missing {what} in line: {line}")))?
        .parse::<T>()
        .map_err(|e| IoError::Mtx(format!("bad {what} '{line}': {e}")))
}

fn read_lines(path: &Path) -> Result<Vec<String>, IoError> {
    let file =
        File::open(path).map_err(|e| IoError::LineFile(format!("open {}: {e}", path.display())))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| IoError::LineFile(format!("read: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.to_string());
    }
    Ok(out)
}

fn read_coords(path: &Path) -> Result<Array2<f32>, IoError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .flexible(false)
        .from_path(path)
        .map_err(|e| IoError::Coords(format!("open {}: {e}", path.display())))?;

    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut n_cols: Option<usize> = None;
    for (row_idx, record) in reader.records().enumerate() {
        let record = record.map_err(|e| IoError::Coords(format!("row {row_idx}: {e}")))?;
        let parsed: Vec<f32> = record
            .iter()
            .map(|s| {
                s.trim()
                    .parse::<f32>()
                    .map_err(|e| IoError::Coords(format!("row {row_idx} value '{s}': {e}")))
            })
            .collect::<Result<_, _>>()?;
        match n_cols {
            None => n_cols = Some(parsed.len()),
            Some(c) if c != parsed.len() => {
                return Err(IoError::Coords(format!(
                    "row {row_idx} has {} columns, expected {}",
                    parsed.len(),
                    c
                )));
            }
            Some(_) => {}
        }
        rows.push(parsed);
    }
    let n_rows = rows.len();
    let n_cols = n_cols.unwrap_or(0);
    if n_cols != 2 && n_cols != 3 {
        return Err(IoError::Coords(format!(
            "expected 2 or 3 columns, got {n_cols}"
        )));
    }

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| IoError::Coords(format!("shape ({n_rows}, {n_cols}) inconsistent: {e}")))
}
