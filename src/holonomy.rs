use serde::{Deserialize, Serialize};

/// Report of holonomy (scheduling drift) detected around a closed loop
/// in the base space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyReport {
    /// Indices forming a closed loop in the base space.
    pub loop_path: Vec<usize>,
    /// Drift magnitude: ‖holonomy matrix − identity‖_F (Frobenius norm).
    pub drift: f64,
    /// Agent IDs affected by this drift.
    pub agent_ids: Vec<String>,
}

impl HolonomyReport {
    /// Compute holonomy for a closed loop given the holonomy matrix.
    pub fn from_matrix(
        loop_path: Vec<usize>,
        holonomy_mat: &[Vec<f64>],
        agent_ids: Vec<String>,
    ) -> Self {
        let drift = frobenius_distance_to_identity(holonomy_mat);
        Self {
            loop_path,
            drift,
            agent_ids,
        }
    }

    /// Whether the drift is considered negligible (below `threshold`).
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.drift < threshold
    }
}

/// Detect all triangle holonomies (i → i+1 → i+2 → i).
///
/// Returns reports for triangles whose drift exceeds `drift_threshold`.
pub fn detect_triangle_holonomies(
    transport_mats: &[Vec<Vec<f64>>],
    n_points: usize,
    drift_threshold: f64,
    agent_ids: &[String],
) -> Vec<HolonomyReport> {
    let dim = transport_mats.first().map(|m| m.len()).unwrap_or(0);
    let mut reports = Vec::new();

    if n_points < 3 || dim == 0 {
        return reports;
    }

    for i in 0..(n_points - 2) {
        let hol = triangle_holonomy(transport_mats, i, dim);
        let drift = frobenius_distance_to_identity(&hol);
        if drift > drift_threshold {
            reports.push(HolonomyReport {
                loop_path: vec![i, i + 1, i + 2, i],
                drift,
                agent_ids: agent_ids.to_vec(),
            });
        }
    }
    reports
}

/// Detect holonomy for an arbitrary closed loop.
pub fn detect_loop_holonomy(
    holonomy_mat: &[Vec<f64>],
    loop_path: Vec<usize>,
    agent_ids: Vec<String>,
    drift_threshold: f64,
) -> Option<HolonomyReport> {
    let report = HolonomyReport::from_matrix(loop_path, holonomy_mat, agent_ids);
    if report.drift > drift_threshold {
        Some(report)
    } else {
        None
    }
}

fn triangle_holonomy(
    transport_mats: &[Vec<Vec<f64>>],
    i: usize,
    dim: usize,
) -> Vec<Vec<f64>> {
    let id = crate::connection::identity_matrix(dim);
    let t01 = transport_mats.get(i).unwrap_or(&id);
    let t12 = transport_mats.get(i + 1).unwrap_or(&id);
    mat_mul(t12, t01)
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

pub(crate) fn frobenius_distance_to_identity(mat: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for (i, row) in mat.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = val - expected;
            sum += diff * diff;
        }
    }
    sum.sqrt()
}
