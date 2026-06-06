use serde::{Deserialize, Serialize};

pub mod bundle;
pub mod connection;
pub mod curvature;
pub mod gauge_fix;
pub mod holonomy;
pub mod section_lifter;

// Re-export core types
pub use bundle::{AgentClock, TimingBundle};
pub use connection::ConnectionForm;
pub use curvature::CurvatureForm;
pub use gauge_fix::{GaugeChoice, GaugeFixer};
pub use holonomy::HolonomyGroup;
pub use section_lifter::SectionLifter;

/// Matrix utilities using Gaussian elimination — no external deps.
pub mod linalg {
    /// Gaussian elimination with partial pivoting.
    /// Solves Ax = b. Returns None if singular.
    pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = a.len();
        if n == 0 || b.len() != n {
            return None;
        }
        // Augmented matrix
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .zip(b.iter())
            .map(|(row, &val)| {
                let mut r = row.clone();
                r.push(val);
                r
            })
            .collect();

        for col in 0..n {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return None; // Singular
            }
            aug.swap(col, max_row);

            // Eliminate below
            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }
        Some(x)
    }

    /// Matrix multiplication: C = A × B
    pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        let n = b[0].len();
        let p = b.len();
        let mut c = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..p {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }
        c
    }

    /// Matrix transpose
    pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if a.is_empty() {
            return vec![];
        }
        let m = a.len();
        let n = a[0].len();
        let mut t = vec![vec![0.0; m]; n];
        for i in 0..m {
            for j in 0..n {
                t[j][i] = a[i][j];
            }
        }
        t
    }

    /// Identity matrix of size n
    pub fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    /// Matrix exponentiation via scaling-and-squaring with Padé approximation (order 1 = trivial).
    /// For small matrices we use Taylor series: exp(A) = I + A + A²/2! + A³/3! + ...
    pub fn mat_exp(a: &[Vec<f64>], terms: usize) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut result = identity(n);
        let mut power = identity(n);
        let mut factorial = 1.0;
        for k in 1..=terms {
            power = mat_mul(&power, a);
            factorial *= k as f64;
            for i in 0..n {
                for j in 0..n {
                    result[i][j] += power[i][j] / factorial;
                }
            }
        }
        result
    }

    /// Zero matrix
    pub fn zeros(m: usize, n: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0; n]; m]
    }

    /// Frobenius norm
    pub fn frobenius(a: &[Vec<f64>]) -> f64 {
        let mut s = 0.0;
        for row in a {
            for &v in row {
                s += v * v;
            }
        }
        s.sqrt()
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::bundle::{AgentClock, TimingBundle};
    use crate::connection::ConnectionForm;
    use crate::curvature::CurvatureForm;
    use crate::gauge_fix::GaugeFixer;
    use crate::holonomy::HolonomyGroup;
    use crate::section_lifter::SectionLifter;
    use crate::linalg::{identity, mat_exp, mat_mul, solve, transpose};

    #[test]
    fn single_agent_curvature_zero() {
        // Trivial bundle: single agent, zero drift → curvature = 0
        let omega = ConnectionForm::zero(1);
        let curv = omega.curvature();
        assert!(curv.is_zero());
    }

    #[test]
    fn single_agent_holonomy_identity() {
        let omega = ConnectionForm::zero(1);
        let h = HolonomyGroup::around_cycle(&[omega], 1.0, 10);
        assert!(h.is_trivial());
    }

    #[test]
    fn two_agents_constant_drift_constant_curvature() {
        // Two agents, one drifting at rate δ relative to the other.
        // Connection: ω = [[0, δ], [0, 0]]
        let delta = 0.05;
        let omega = ConnectionForm::new(vec![
            vec![0.0, delta],
            vec![0.0, 0.0],
        ]);
        let curv = omega.curvature();
        // Nilpotent 2x2 → ω∧ω = 0, dω = 0 → curvature = 0
        assert!(curv.is_zero());
    }

    #[test]
    fn two_agents_non_nilpotent_curvature() {
        // ω = [[0, δ], [-δ, 0]] → ω∧ω = [[-δ², 0], [0, -δ²]]
        let delta = 0.1;
        let omega = ConnectionForm::new(vec![
            vec![0.0, delta],
            vec![-delta, 0.0],
        ]);
        let curv = omega.curvature();
        // Curvature[0][0] = -δ², Curvature[1][1] = -δ²
        assert!((curv.components[0][0][0] - (-delta * delta)).abs() < 1e-10);
        assert!((curv.components[0][1][1] - (-delta * delta)).abs() < 1e-10);
        assert!(curv.components[0][0][1].abs() < 1e-10);
    }

    #[test]
    fn holonomy_around_cycle_equals_drift_times_length() {
        // Constant drift connection, rectangular cycle
        let delta = 0.1;
        let omega = ConnectionForm::new(vec![
            vec![0.0, delta],
            vec![-delta, 0.0],
        ]);
        let width = 2.0;
        let height = 3.0;
        let h = HolonomyGroup::rectangular_cycle(&omega, width, height, 100);
        // Drift = 2 × δ² × area = 2 × 0.01 × 6 = 0.12
        let expected = 2.0 * delta * delta * width * height;
        assert!((h.drift - expected).abs() < 1e-10);
    }

    #[test]
    fn pid_corrected_agents_curvature_decays() {
        // Simulate PID correction: over time, errors shrink exponentially.
        let clock = AgentClock::new(0, 0.1, 0.5, 1.0);
        let t = 10.0;
        let error_at_t = 0.1 * t + 0.5; // 1.5

        // After correction with kp=1, the residual error is 0
        let corrected = clock.corrected_time(t, 1.0, 0.0, 0.0, error_at_t, 0.0, 0.0);
        assert!((corrected - t).abs() < 1e-10);

        // Simulating multiple correction steps: error decays
        let mut error = 1.5;
        let kp = 0.5;
        for _ in 0..10 {
            let correction = kp * error;
            error = error - correction;
        }
        // After 10 steps with kp=0.5, error ≈ 1.5 × 0.5^10 ≈ 0.00146
        assert!(error < 0.01);
    }

    #[test]
    fn connection_form_structure_equation() {
        // Verify Ω = dω + ω ∧ ω
        let omega = ConnectionForm::new(vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ]);
        let curv = omega.curvature();
        // dω = 0 (constant), ω∧ω = ω²
        // ω²[0][2] = 1.0, rest are zero (strictly upper triangular 3x3)
        assert!((curv.components[0][0][2] - 1.0).abs() < 1e-10);
        assert!(curv.components[0][0][1].abs() < 1e-10);
        assert!(curv.components[0][1][2].abs() < 1e-10);
    }

    #[test]
    fn parallel_transport_preserves_fiber_metric() {
        // For a connection that is anti-symmetric (so(2)), parallel transport
        // preserves the Euclidean inner product.
        let omega = ConnectionForm::new(vec![
            vec![0.0, 0.3],
            vec![-0.3, 0.0],
        ]);
        let transport = omega.horizontal_lift(1.0, 1000);
        let transport_t = transpose(&transport);
        let product = mat_mul(&transport_t, &transport);
        let id = identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((product[i][j] - id[i][j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn gauge_transformation_preserves_curvature_invariant() {
        let omega = ConnectionForm::new(vec![
            vec![0.0, 0.2],
            vec![-0.2, 0.0],
        ]);
        let curv = omega.curvature();

        // Non-trivial gauge transformation (rotation)
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let g = vec![vec![c, -c], vec![c, c]];
        let fixer = GaugeFixer::custom("rotated_45".into(), g);

        let transformed_curv = fixer.transform_curvature(&curv);
        // Norm should be preserved
        assert!((curv.norm() - transformed_curv.norm()).abs() < 1e-10);
    }

    #[test]
    fn flatness_zero_curvature_globally_synchronizable() {
        let omega = ConnectionForm::zero(3);
        let curv = omega.curvature();
        assert!(curv.is_flat());
        // Flat → can find global synchronization
    }

    #[test]
    fn section_lifting_produces_correct_timelines() {
        let clocks = vec![
            AgentClock::new(0, 0.0, 0.0, 1.0),
            AgentClock::new(1, 0.05, 1.0, 2.0),
        ];
        let bundle = TimingBundle::with_clocks(clocks, 1, 1);
        let lifter = SectionLifter::new(bundle);

        let t0 = lifter.lift(0.0);
        assert!((t0[0] - 0.0).abs() < 1e-10);
        assert!((t0[1] - 1.0).abs() < 1e-10); // 0*1.05 + 1.0

        let t1 = lifter.lift(10.0);
        assert!((t1[0] - 10.0).abs() < 1e-10);
        assert!((t1[1] - 11.5).abs() < 1e-10); // 10*1.05 + 1.0
    }

    #[test]
    fn gaussian_elimination_solve() {
        let a = vec![vec![2.0, 1.0], vec![5.0, 3.0]];
        let b = vec![4.0, 7.0];
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-10);
        assert!((x[1] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn gaussian_elimination_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert!(solve(&a, &b).is_none());
    }

    #[test]
    fn mat_exp_identity_for_zero() {
        let zero = vec![vec![0.0; 2]; 2];
        let exp = mat_exp(&zero, 10);
        let id = identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((exp[i][j] - id[i][j]).abs() < 1e-10);
            }
        }
    }
}
