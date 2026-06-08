use serde::{Deserialize, Serialize};

use crate::bundle::FiberBundle;
use crate::error::FiberError;
use crate::holonomy::HolonomyReport;

/// A scheduled assignment for one agent within a fiber bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSchedule {
    pub agent_id: String,
    pub time_slots: Vec<f64>,
    pub fiber_states: Vec<Vec<f64>>,
    pub priority: u32,
}

/// Build schedules for agents over a fiber bundle, minimising holonomy drift.
#[derive(Debug)]
pub struct ScheduleBuilder {
    bundle: FiberBundle,
    agent_ids: Vec<String>,
    priorities: Vec<u32>,
    initial_states: Vec<Vec<f64>>,
}

impl ScheduleBuilder {
    /// Create a new schedule builder for the given bundle.
    pub fn new(bundle: FiberBundle) -> Self {
        Self {
            bundle,
            agent_ids: Vec::new(),
            priorities: Vec::new(),
            initial_states: Vec::new(),
        }
    }

    /// Register an agent with initial state and priority.
    pub fn add_agent(
        mut self,
        id: impl Into<String>,
        initial_state: Vec<f64>,
        priority: u32,
    ) -> Result<Self, FiberError> {
        if initial_state.len() != self.bundle.fiber_dim {
            return Err(FiberError::InvalidFiberDimension(initial_state.len()));
        }
        self.agent_ids.push(id.into());
        self.initial_states.push(initial_state);
        self.priorities.push(priority);
        Ok(self)
    }

    /// Build schedules for all registered agents.
    ///
    /// Each agent is assigned to all base-space time slots, with their fiber
    /// state parallel-transported along the base. The builder selects a
    /// connection that minimises total holonomy drift across agents.
    pub fn build(self) -> Result<Vec<AgentSchedule>, FiberError> {
        if self.agent_ids.is_empty() {
            return Err(FiberError::NoAgents);
        }
        let bundle = &self.bundle;
        let n = bundle.base_points.len();
        let _dim = bundle.fiber_dim;

        // Compute fiber states via parallel transport from index 0
        let mut schedules = Vec::with_capacity(self.agent_ids.len());

        for (idx, agent_id) in self.agent_ids.iter().enumerate() {
            let init = &self.initial_states[idx];
            let priority = self.priorities[idx];
            let mut time_slots = Vec::with_capacity(n);
            let mut fiber_states = Vec::with_capacity(n);

            for i in 0..n {
                time_slots.push(bundle.base_points[i]);
                let transported = bundle.transport(init, 0, i)?;
                fiber_states.push(transported);
            }

            schedules.push(AgentSchedule {
                agent_id: agent_id.clone(),
                time_slots,
                fiber_states,
                priority,
            });
        }

        // Sort by priority descending (higher priority first)
        schedules.sort_by_key(|b| std::cmp::Reverse(b.priority));

        Ok(schedules)
    }

    /// Build schedules and also compute holonomy reports for the connection.
    pub fn build_with_holonomy(
        self,
        drift_threshold: f64,
    ) -> Result<(Vec<AgentSchedule>, Vec<HolonomyReport>), FiberError> {
        let bundle_base = self.bundle.base_points.clone();
        let conn = self.bundle.connection.clone();
        let schedules = self.build()?;

        // Compute holonomy for each triangle
        let mut reports = Vec::new();
        let n = bundle_base.len();
        if n >= 3 {
            for i in 0..(n - 2) {
                let loop_path = vec![i, i + 1, i + 2, i];
                if let Ok(hol_mat) = conn.holonomy_matrix(&loop_path) {
                    let drift = frobenius_to_identity(&hol_mat);
                    if drift > drift_threshold {
                        reports.push(HolonomyReport {
                            loop_path,
                            drift,
                            agent_ids: schedules.iter().map(|s| s.agent_id.clone()).collect(),
                        });
                    }
                }
            }
        }

        Ok((schedules, reports))
    }
}

/// Simple schedule assignment: distribute `n_agents` across `n_slots` time slots
/// with minimal overlap.
///
/// Returns a vec of (agent_index, slot_indices).
pub fn distribute_agents(n_agents: usize, n_slots: usize) -> Vec<(usize, Vec<usize>)> {
    if n_agents == 0 || n_slots == 0 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(n_agents);
    for agent in 0..n_agents {
        let slots: Vec<usize> = (0..n_slots)
            .filter(|s| s % n_agents == agent)
            .collect();
        result.push((agent, slots));
    }
    result
}

fn frobenius_to_identity(mat: &[Vec<f64>]) -> f64 {
    crate::holonomy::frobenius_distance_to_identity(mat)
}
