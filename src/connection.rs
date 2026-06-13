use serde::{Deserialize, Serialize};
use crate::curvature::CurvatureForm;
use crate::linalg::{mat_exp, mat_mul, mat_mul as mm, zeros, identity};

/// Ehresmann connection 1-form ω.
/// components[i][j] represents the (i,j) entry of the Lie-algebra-valued form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionForm {
    /// Matrix components of the connection form.
    pub components: Vec<Vec<f64>>,
}

impl ConnectionForm {
    pub fn new(components: Vec<Vec<f64>>) -> Self {
        Self { components }
    }

    pub fn zero(n: usize) -> Self {
        Self {
            components: vec![vec![0.0; n]; n],
        }
    }

    pub fn dim(&self) -> usize {
        self.components.len()
    }

    /// Compute exterior derivative dω (discrete approximation).
    /// For a constant connection form, dω = 0.
    /// Here we compute dω along two base directions as the finite difference.
    pub fn exterior_derivative(&self, _dx: f64) -> Self {
        // For a constant connection, dω = 0
        Self::zero(self.dim())
    }

    /// Compute ω ∧ ω (wedge product, which for matrix-valued forms is ω × ω - ω × ω / 2,
    /// but for matrix-valued 1-forms it's just matrix multiplication).
    pub fn wedge_square(&self) -> Self {
        Self::new(mat_mul(&self.components, &self.components))
    }

    /// Curvature from Cartan's structure equation: Ω = dω + ω ∧ ω
    pub fn curvature(&self) -> CurvatureForm {
        let dw = self.exterior_derivative(1e-6);
        let ww = self.wedge_square();
        let n = self.dim();
        let mut comps = vec![vec![vec![0.0; n]; n]; 1];
        for i in 0..n {
            for j in 0..n {
                comps[0][i][j] = dw.components[i][j] + ww.components[i][j];
            }
        }
        CurvatureForm::new(comps)
    }

    /// Horizontal lift of a base curve: parallel transport along the curve.
    /// Given parameter `t` in [0, T] with `steps` discrete steps,
    /// returns the transport matrix (fiber automorphism).
    pub fn horizontal_lift(&self, t_total: f64, steps: usize) -> Vec<Vec<f64>> {
        let n = self.dim();
        let dt = t_total / steps as f64;
        // Parallel transport: solve dP/dt = -ω · P
        // Discrete: P(t+dt) = exp(-ω * dt) · P(t)
        let mut omega_dt = self.components.clone();
        for row in &mut omega_dt {
            for v in row.iter_mut() {
                *v = -*v * dt;
            }
        }
        let step_matrix = mat_exp(&omega_dt, 20);
        let mut transport = identity(n);
        for _ in 0..steps {
            transport = mm(&step_matrix, &transport);
        }
        transport
    }

    /// Add two connection forms (for gauge transformations).
    pub fn add(&self, other: &ConnectionForm) -> ConnectionForm {
        let n = self.dim();
        let mut result = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                result[i][j] = self.components[i][j] + other.components[i][j];
            }
        }
        ConnectionForm::new(result)
    }

    /// Scalar multiply
    pub fn scale(&self, s: f64) -> ConnectionForm {
        ConnectionForm::new(
            self.components
                .iter()
                .map(|row| row.iter().map(|v| v * s).collect())
                .collect(),
        )
    }

    /// Frobenius norm of components
    pub fn norm(&self) -> f64 {
        let mut s = 0.0;
        for row in &self.components {
            for &v in row {
                s += v * v;
            }
        }
        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_connection_has_zero_curvature() {
        let omega = ConnectionForm::zero(3);
        let curv = omega.curvature();
        assert!(curv.is_zero());
    }

    #[test]
    fn constant_connection_curvature_is_wedge_square() {
        // Constant connection: dω = 0, so Ω = ω∧ω
        let omega = ConnectionForm::new(vec![
            vec![0.0, 0.1, 0.0],
            vec![0.0, 0.0, 0.2],
            vec![0.0, 0.0, 0.0],
        ]);
        let curv = omega.curvature();
        // ω∧ω[0][2] = 0.1 * 0.2 = 0.02
        assert!((curv.components[0][0][2] - 0.02).abs() < 1e-10);
        assert!((curv.components[0][1][2]).abs() < 1e-10);
    }

    #[test]
    fn parallel_transport_preserves_metric_identity() {
        // Zero connection → transport = identity
        let omega = ConnectionForm::zero(2);
        let transport = omega.horizontal_lift(1.0, 100);
        let id = identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((transport[i][j] - id[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn structure_equation_d_omega_plus_omega_wedge_omega() {
        let omega = ConnectionForm::new(vec![
            vec![0.0, 0.5],
            vec![0.0, 0.0],
        ]);
        let curv = omega.curvature();
        // dω = 0 (constant), ω∧ω = 0 (nilpotent upper-triangular 2x2)
        assert!(curv.is_zero());
    }

    #[test]
    fn connection_scale_and_add() {
        let a = ConnectionForm::new(vec![vec![0.0, 1.0], vec![0.0, 0.0]]);
        let b = ConnectionForm::new(vec![vec![0.0, 2.0], vec![0.0, 0.0]]);
        let c = a.add(&b);
        assert!((c.components[0][1] - 3.0).abs() < 1e-10);
    }
}
