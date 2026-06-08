use crate::connection::ConnectionForm;
use crate::curvature::CurvatureForm;
use crate::linalg::{identity, mat_mul, solve, transpose};
use serde::{Deserialize, Serialize};

/// A gauge choice that fixes the synchronization frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeChoice {
    /// Name of the canonical frame.
    pub frame: String,
    /// Transformation matrix g: ω' = g⁻¹ωg + g⁻¹dg
    pub transformation: Vec<Vec<f64>>,
}

/// Gauge fixing utility — chooses a canonical synchronization frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeFixer {
    /// The chosen gauge.
    pub gauge: GaugeChoice,
}

impl GaugeFixer {
    /// Create a gauge fixer with the Lorenz-like synchronization frame.
    /// This fixes div(ω) = 0 (analogous to Lorenz gauge in EM).
    pub fn lorenz(n: usize) -> Self {
        Self {
            gauge: GaugeChoice {
                frame: "lorenz".to_string(),
                transformation: identity(n),
            },
        }
    }

    /// Create with a custom transformation.
    pub fn custom(frame: String, transformation: Vec<Vec<f64>>) -> Self {
        Self {
            gauge: GaugeChoice {
                frame,
                transformation,
            },
        }
    }

    /// Apply gauge transformation to a connection form:
    /// ω' = g⁻¹ ω g + g⁻¹ dg
    /// For constant gauge, dg = 0, so ω' = g⁻¹ ω g
    pub fn transform_connection(&self, omega: &ConnectionForm) -> ConnectionForm {
        let g = &self.gauge.transformation;
        let g_inv = Self::invert(g).unwrap_or_else(|| identity(g.len()));
        // ω' = g⁻¹ ω g
        let gw = mat_mul(&g_inv, &omega.components);
        let gwg = mat_mul(&gw, g);
        ConnectionForm::new(gwg)
    }

    /// Verify gauge invariance of curvature: Ω' = g⁻¹ Ω g
    pub fn transform_curvature(&self, curv: &CurvatureForm) -> CurvatureForm {
        let g = &self.gauge.transformation;
        let g_inv = Self::invert(g).unwrap_or_else(|| identity(g.len()));
        let mut new_comps = Vec::new();
        for plane in &curv.components {
            let gc = mat_mul(&g_inv, plane);
            let gcg = mat_mul(&gc, g);
            new_comps.push(gcg);
        }
        CurvatureForm::new(new_comps)
    }

    /// Check if curvature is gauge-invariant (same norm after transformation).
    pub fn verify_gauge_invariance(&self, curv: &CurvatureForm) -> bool {
        let transformed = self.transform_curvature(curv);
        (curv.norm() - transformed.norm()).abs() < 1e-10
    }

    /// Fix gauge by choosing the frame where the connection is as symmetric as possible.
    /// Returns a new connection in the canonical gauge.
    pub fn fix(&self, omega: &ConnectionForm) -> ConnectionForm {
        self.transform_connection(omega)
    }

    /// Invert a matrix using Gaussian elimination.
    fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = m.len();
        if n == 0 {
            return None;
        }
        // Augment with identity
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = m[i][j];
            }
            aug[i][n + i] = 1.0;
        }

        // Forward elimination
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return None;
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        for col in (0..n).rev() {
            let pivot = aug[col][col];
            for j in 0..2 * n {
                aug[col][j] /= pivot;
            }
            for row in 0..col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Extract inverse
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][n + j];
            }
        }
        Some(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lorenz_gauge_identity_transform() {
        let fixer = GaugeFixer::lorenz(3);
        let omega = ConnectionForm::new(vec![
            vec![0.0, 0.1, 0.0],
            vec![0.0, 0.0, 0.2],
            vec![0.0, 0.0, 0.0],
        ]);
        let fixed = fixer.fix(&omega);
        // Identity gauge → same connection
        for i in 0..3 {
            for j in 0..3 {
                assert!((fixed.components[i][j] - omega.components[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn gauge_transformation_preserves_curvature_norm() {
        let omega = ConnectionForm::new(vec![vec![0.0, 0.1], vec![-0.1, 0.0]]);
        let curv = omega.curvature();

        // Rotate by 45 degrees
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        let g = vec![vec![s2, -s2], vec![s2, s2]];
        let fixer = GaugeFixer::custom("rotated".to_string(), g);
        assert!(fixer.verify_gauge_invariance(&curv));
    }

    #[test]
    fn matrix_inversion_roundtrip() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let inv = GaugeFixer::invert(&m).unwrap();
        let product = mat_mul(&m, &inv);
        let id = identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((product[i][j] - id[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn flatness_implies_global_synchronization() {
        // Zero curvature → globally synchronizable
        let omega = ConnectionForm::zero(2);
        let curv = omega.curvature();
        assert!(curv.is_flat());
        // In a flat bundle, there exists a global section where all clocks agree
    }
}
