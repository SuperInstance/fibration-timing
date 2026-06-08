use serde::{Deserialize, Serialize};

use crate::connection::Connection;
use crate::error::FiberError;

/// A fiber bundle (E, B, π, F) modelling concurrent agent scheduling.
///
/// - **Base space B** = timeline (`base_points`).
/// - **Fiber F** = agent state space of dimension `fiber_dim`.
/// - **Total space E** = all (time, state) pairs.
/// - **Connection** = defines parallel transport (time evolution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiberBundle {
    /// Timeline points (base space B ⊂ ℝ).
    pub base_points: Vec<f64>,
    /// Dimension of the fiber (agent state space).
    pub fiber_dim: usize,
    /// Connection defining parallel transport.
    pub connection: Connection,
}

impl FiberBundle {
    /// Create a new fiber bundle over a sorted timeline with given fiber dimension.
    ///
    /// The connection is initialised as flat (identity transport).
    pub fn new(base_points: Vec<f64>, fiber_dim: usize) -> Result<Self, FiberError> {
        if base_points.is_empty() {
            return Err(FiberError::EmptyBaseSpace);
        }
        if fiber_dim == 0 {
            return Err(FiberError::InvalidFiberDimension(0));
        }
        let segments = base_points.len().saturating_sub(1).max(1);
        let connection = Connection::flat(fiber_dim, segments);
        Ok(Self {
            base_points,
            fiber_dim,
            connection,
        })
    }

    /// Create a bundle with a custom connection.
    pub fn with_connection(
        base_points: Vec<f64>,
        fiber_dim: usize,
        connection: Connection,
    ) -> Result<Self, FiberError> {
        if base_points.is_empty() {
            return Err(FiberError::EmptyBaseSpace);
        }
        if fiber_dim == 0 {
            return Err(FiberError::InvalidFiberDimension(0));
        }
        // Validate transport matrix dimensions
        for (i, mat) in connection.transport.iter().enumerate() {
            if mat.len() != fiber_dim || mat.iter().any(|row| row.len() != fiber_dim) {
                return Err(FiberError::TransportMatrixSizeMismatch {
                    expected: fiber_dim,
                    got: mat.len(),
                });
            }
            let _ = i; // suppress unused warning
        }
        Ok(Self {
            base_points,
            fiber_dim,
            connection,
        })
    }

    /// Number of base points (timeline length).
    pub fn len(&self) -> usize {
        self.base_points.len()
    }

    /// Whether the base space is empty (should never happen after construction).
    pub fn is_empty(&self) -> bool {
        self.base_points.is_empty()
    }

    /// Projection π: given an index into the base space, return the time point.
    pub fn projection(&self, idx: usize) -> Result<f64, FiberError> {
        self.base_points
            .get(idx)
            .copied()
            .ok_or(FiberError::IndexOutOfBounds {
                index: idx,
                len: self.base_points.len(),
            })
    }

    /// Find the base-space index closest to the given time.
    pub fn nearest_index(&self, t: f64) -> usize {
        let mut best = 0;
        let mut best_dist = (self.base_points[0] - t).abs();
        for (i, &pt) in self.base_points.iter().enumerate().skip(1) {
            let d = (pt - t).abs();
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        best
    }

    /// Parallel-transport a fiber vector from index `from` to index `to`.
    pub fn transport(
        &self,
        fiber_state: &[f64],
        from: usize,
        to: usize,
    ) -> Result<Vec<f64>, FiberError> {
        self.connection.transport_vector(fiber_state, from, to)
    }
}
