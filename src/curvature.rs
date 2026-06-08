use crate::linalg::frobenius;
use serde::{Deserialize, Serialize};

/// Curvature 2-form Ω = dω + ω ∧ ω.
/// For multi-dimensional base, components[k][i][j] is the (i,j) component
/// of the curvature in the k-th 2-form slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureForm {
    pub components: Vec<Vec<Vec<f64>>>,
}

impl CurvatureForm {
    pub fn new(components: Vec<Vec<Vec<f64>>>) -> Self {
        Self { components }
    }

    pub fn zero(n: usize) -> Self {
        Self {
            components: vec![vec![vec![0.0; n]; n]; 1],
        }
    }

    pub fn dim(&self) -> usize {
        if self.components.is_empty() {
            0
        } else {
            self.components[0].len()
        }
    }

    pub fn is_zero(&self) -> bool {
        self.norm() < 1e-12
    }

    /// Total Frobenius norm of all components.
    pub fn norm(&self) -> f64 {
        let mut s = 0.0;
        for plane in &self.components {
            s += frobenius(plane);
        }
        s
    }

    /// Check flatness: zero curvature ↔ globally synchronizable.
    pub fn is_flat(&self) -> bool {
        self.is_zero()
    }

    /// Compute cumulative desynchronization over a dialogue exchange.
    /// Given `exchanges` units of dialogue, total drift = curvature magnitude × exchanges.
    pub fn desynchronization(&self, exchanges: f64) -> f64 {
        self.norm() * exchanges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_curvature_is_flat() {
        let c = CurvatureForm::zero(3);
        assert!(c.is_flat());
        assert!(c.is_zero());
    }

    #[test]
    fn nonzero_curvature_not_flat() {
        let c = CurvatureForm::new(vec![vec![vec![0.0, 0.1], vec![0.0, 0.0]]]);
        assert!(!c.is_flat());
    }

    #[test]
    fn desynchronization_scales_linearly() {
        let c = CurvatureForm::new(vec![vec![vec![0.0, 0.5], vec![0.0, 0.0]]]);
        let d1 = c.desynchronization(1.0);
        let d2 = c.desynchronization(2.0);
        assert!((d2 - 2.0 * d1).abs() < 1e-10);
    }

    #[test]
    fn curvature_norm_matches_frobenius() {
        let c = CurvatureForm::new(vec![vec![vec![3.0, 4.0], vec![0.0, 0.0]]]);
        // Frobenius = sqrt(9 + 16) = 5.0
        assert!((c.norm() - 5.0).abs() < 1e-10);
    }
}
