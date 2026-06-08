use serde::{Deserialize, Serialize};

use crate::error::FiberError;

/// A connection on a fiber bundle defines parallel transport between fibers
/// and encodes curvature (the failure of transport to commute).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Parallel transport matrices. `transport[i]` is the d×d matrix
    /// transporting from base point i to i+1 (or from point i to i if singleton).
    pub transport: Vec<Vec<Vec<f64>>>,
}

impl Connection {
    /// Create a flat (identity) connection of the given dimension over `n` segments.
    ///
    /// A flat connection has zero curvature: transport around any loop returns
    /// the identity, i.e. no scheduling drift.
    pub fn flat(dim: usize, segments: usize) -> Self {
        let mat = identity_matrix(dim);
        Self {
            transport: vec![mat; segments.max(1)],
        }
    }

    /// Create a connection with small random-ish perturbations for testing.
    ///
    /// Each transport matrix is identity + a small skew perturbation,
    /// modelling non-trivial curvature (agent drift).
    pub fn perturbed(dim: usize, segments: usize, seed: u64) -> Self {
        let mut mats = Vec::with_capacity(segments);
        // Simple LCG for deterministic pseudo-random perturbation
        let mut state = seed;
        for _ in 0..segments {
            let mut mat = identity_matrix(dim);
            for (i, row) in mat.iter_mut().enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    if i != j {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let raw = ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
                        *val = raw * 0.1;
                    }
                }
            }
            mats.push(mat);
        }
        Self { transport: mats }
    }

    /// Parallel-transport a fiber vector from base index `from` to `to`.
    ///
    /// Returns the transported vector.
    pub fn transport_vector(
        &self,
        v: &[f64],
        from: usize,
        to: usize,
    ) -> Result<Vec<f64>, FiberError> {
        let n = self.transport.len();
        if n == 0 {
            return Ok(v.to_vec());
        }
        let mut result = v.to_vec();

        if from <= to {
            for i in from..to {
                let mat = self.get_matrix(i)?;
                result = mat_vec_mul(mat, &result);
            }
        } else {
            // Reverse transport
            for i in (to..from).rev() {
                let mat = self.get_matrix(i)?;
                result = mat_vec_mul(mat, &result);
            }
        }
        Ok(result)
    }

    /// Compute the holonomy (round-trip transport matrix) for a closed loop.
    ///
    /// Returns the product of transport matrices around the loop.
    pub fn holonomy_matrix(&self, loop_path: &[usize]) -> Result<Vec<Vec<f64>>, FiberError> {
        if loop_path.len() < 2 {
            return Err(FiberError::LoopTooShort(loop_path.len()));
        }
        let first = loop_path[0];
        let last = *loop_path.last().unwrap();
        if first != last {
            return Err(FiberError::LoopNotClosed {
                start: first,
                end: last,
            });
        }
        let dim = self
            .transport
            .first()
            .map(|m| m.len())
            .unwrap_or(0)
            .max(1);
        let mut accum = identity_matrix(dim);

        for window in loop_path.windows(2) {
            let a = window[0];
            let b = window[1];
            if a == b {
                continue;
            }
            if a < b {
                for i in a..b {
                    let mat = self.get_matrix(i)?;
                    accum = mat_mul(&accum, mat);
                }
            } else {
                for i in (b..a).rev() {
                    let mat = self.get_matrix(i)?;
                    accum = mat_mul(&accum, mat);
                }
            }
        }
        Ok(accum)
    }

    /// Compute the curvature tensor between three consecutive base points.
    ///
    /// Curvature = holonomy around the triangle (i, i+1, i+2, i).
    pub fn curvature_triangle(&self, i: usize) -> Result<Vec<Vec<f64>>, FiberError> {
        let loop_path = vec![i, i + 1, i + 2, i];
        self.holonomy_matrix(&loop_path)
    }

    fn get_matrix(&self, idx: usize) -> Result<&Vec<Vec<f64>>, FiberError> {
        self.transport.get(idx).ok_or(FiberError::IndexOutOfBounds {
            index: idx,
            len: self.transport.len(),
        })
    }
}

// ---- helpers ----

pub(crate) fn identity_matrix(dim: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; dim]; dim];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let dim = v.len();
    let mut out = vec![0.0; dim];
    for (i, out_val) in out.iter_mut().enumerate() {
        for j in 0..dim {
            *out_val += mat[i][j] * v[j];
        }
    }
    out
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for (i, row) in c.iter_mut().enumerate() {
        for j in 0..m {
            for k in 0..p {
                row[j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}
