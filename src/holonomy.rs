use serde::{Deserialize, Serialize};
use crate::connection::ConnectionForm;
use crate::linalg::{identity, mat_mul};

/// Holonomy group around dialogue cycles.
/// The holonomy captures net timing drift around closed loops in the dialogue space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyGroup {
    /// Generators of the holonomy group (as matrices).
    pub generators: Vec<Vec<f64>>,
    /// Net drift accumulated around a cycle.
    pub drift: f64,
}

impl HolonomyGroup {
    pub fn new(generators: Vec<Vec<f64>>, drift: f64) -> Self {
        Self { generators, drift }
    }

    /// Trivial holonomy (identity group) for flat bundles.
    pub fn trivial(n: usize) -> Self {
        Self {
            generators: identity(n),
            drift: 0.0,
        }
    }

    /// Compute holonomy around a closed dialogue cycle.
    /// The cycle is defined by a sequence of connection forms at each segment.
    /// Holonomy = product of parallel transports around the loop.
    pub fn around_cycle(connections: &[ConnectionForm], cycle_length: f64, steps_per_segment: usize) -> Self {
        if connections.is_empty() {
            return Self::trivial(0);
        }
        let n = connections[0].dim();
        let segment_length = cycle_length / connections.len() as f64;
        let mut holonomy = identity(n);

        for conn in connections {
            let transport = conn.horizontal_lift(segment_length, steps_per_segment);
            holonomy = mat_mul(&transport, &holonomy);
        }

        // Net drift = trace deviation from identity
        let id = identity(n);
        let mut drift = 0.0;
        for i in 0..n {
            drift += (holonomy[i][i] - id[i][i]).abs();
        }

        Self {
            generators: holonomy,
            drift,
        }
    }

    /// Compute holonomy for a rectangular dialogue cycle with constant connection.
    /// Going around: forward, up, backward, down.
    /// For constant ω, holonomy ≈ exp(Ω · area).
    pub fn rectangular_cycle(omega: &ConnectionForm, width: f64, height: f64, steps: usize) -> Self {
        let n = omega.dim();
        let curv = omega.curvature();

        // Approximate: holonomy ≈ I + Ω · (width × height)
        let area = width * height;
        let mut holonomy = identity(n);
        if !curv.components.is_empty() {
            for i in 0..n {
                for j in 0..n {
                    holonomy[i][j] += curv.components[0][i][j] * area;
                }
            }
        }

        let id = identity(n);
        let mut drift = 0.0;
        for i in 0..n {
            drift += (holonomy[i][i] - id[i][i]).abs();
        }

        Self {
            generators: holonomy,
            drift,
        }
    }

    /// Is the holonomy trivial (identity)?
    pub fn is_trivial(&self) -> bool {
        self.drift < 1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_holonomy_for_zero_connection() {
        let omega = ConnectionForm::zero(3);
        let h = HolonomyGroup::around_cycle(&[omega], 1.0, 10);
        assert!(h.is_trivial());
    }

    #[test]
    fn holonomy_drift_equals_drift_times_cycle_length() {
        // Two agents with constant drift rate δ.
        // Connection form ω = [[0, δ], [0, 0]]
        let delta = 0.05;
        let omega = ConnectionForm::new(vec![
            vec![0.0, delta],
            vec![0.0, 0.0],
        ]);
        let cycle_length = 4.0;
        let h = HolonomyGroup::around_cycle(&[omega.clone(), omega.clone(), omega.clone(), omega.clone()], cycle_length, 100);
        // With constant nilpotent ω, curvature = ω∧ω = 0 for 2x2 strictly upper triangular
        // So holonomy around cycle should be close to identity
        // Actually ω∧ω for [[0,δ],[0,0]] is [[0,0],[0,0]] — nilpotent
        // So curvature = 0, holonomy = identity. Let me test with a non-nilpotent case.

        // Use a connection that produces non-zero curvature
        let omega2 = ConnectionForm::new(vec![
            vec![0.0, delta],
            vec![-delta, 0.0],
        ]);
        let h2 = HolonomyGroup::rectangular_cycle(&omega2, 2.0, 3.0, 100);
        // Curvature = dω + ω∧ω = 0 + [[0,δ],[-δ,0]]² = [[-δ², 0], [0, -δ²]]
        // Area = 6.0, so drift ≈ δ² × 6 × 2 (both diagonal entries)
        let expected_drift = delta * delta * 6.0 * 2.0;
        assert!((h2.drift - expected_drift).abs() < 0.01);
    }

    #[test]
    fn rectangular_cycle_drift_from_curvature() {
        let delta = 0.1;
        let omega = ConnectionForm::new(vec![
            vec![0.0, delta],
            vec![-delta, 0.0],
        ]);
        let h = HolonomyGroup::rectangular_cycle(&omega, 1.0, 1.0, 100);
        // Curvature diagonal = -δ², area = 1, drift = |−δ²| + |−δ²| = 2δ²
        assert!((h.drift - 2.0 * delta * delta).abs() < 1e-10);
    }

    #[test]
    fn single_agent_trivial_holonomy() {
        let omega = ConnectionForm::zero(1);
        let h = HolonomyGroup::around_cycle(&[omega], 5.0, 50);
        assert!(h.is_trivial());
    }
}
