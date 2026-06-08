use serde::{Deserialize, Serialize};

/// An agent's local clock with timing characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentClock {
    pub id: usize,
    /// Clock drift rate (fractional, per unit base time).
    pub drift_rate: f64,
    /// Fixed latency offset.
    pub latency: f64,
    /// Rhythm period — natural response cadence.
    pub rhythm: f64,
}

impl AgentClock {
    pub fn new(id: usize, drift_rate: f64, latency: f64, rhythm: f64) -> Self {
        Self {
            id,
            drift_rate,
            latency,
            rhythm,
        }
    }

    /// Compute the local time at base time `t`.
    pub fn local_time(&self, t: f64) -> f64 {
        t * (1.0 + self.drift_rate) + self.latency
    }

    /// Compute local time with PID correction applied.
    /// The PID controller targets zero desynchronization.
    pub fn corrected_time(
        &self,
        t: f64,
        kp: f64,
        ki: f64,
        kd: f64,
        error: f64,
        integral: f64,
        derivative: f64,
    ) -> f64 {
        let correction = kp * error + ki * integral + kd * derivative;
        self.local_time(t) - correction
    }
}

/// Principal timing bundle: base = shared timeline, fiber = product of agent clocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBundle {
    /// Dimension of the base manifold (shared timeline parameters).
    pub base_dim: usize,
    /// Dimension of the fiber (agent timing state per agent).
    pub fiber_dim: usize,
    /// Number of agents.
    pub agents: usize,
    /// The clocks for each agent.
    pub clocks: Vec<AgentClock>,
}

impl TimingBundle {
    pub fn new(base_dim: usize, fiber_dim: usize, agents: usize) -> Self {
        let clocks = (0..agents)
            .map(|id| AgentClock::new(id, 0.0, 0.0, 1.0))
            .collect();
        Self {
            base_dim,
            fiber_dim,
            agents,
            clocks,
        }
    }

    pub fn with_clocks(clocks: Vec<AgentClock>, base_dim: usize, fiber_dim: usize) -> Self {
        let agents = clocks.len();
        Self {
            base_dim,
            fiber_dim,
            agents,
            clocks,
        }
    }

    /// Total fiber dimension = agents × fiber_dim.
    pub fn total_fiber_dim(&self) -> usize {
        self.agents * self.fiber_dim
    }

    /// Trivial bundle check: single agent with zero drift → bundle is a product.
    pub fn is_trivial(&self) -> bool {
        self.agents == 1 && self.clocks[0].drift_rate.abs() < 1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_bundle_single_agent() {
        let bundle = TimingBundle::new(1, 1, 1);
        assert!(bundle.is_trivial());
        assert_eq!(bundle.total_fiber_dim(), 1);
    }

    #[test]
    fn agent_clock_local_time() {
        let clock = AgentClock::new(0, 0.1, 0.5, 1.0);
        // At t=10: local = 10 * 1.1 + 0.5 = 11.5
        assert!((clock.local_time(10.0) - 11.5).abs() < 1e-10);
    }

    #[test]
    fn zero_drift_clock() {
        let clock = AgentClock::new(0, 0.0, 0.0, 1.0);
        assert!((clock.local_time(5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn pid_corrected_time() {
        let clock = AgentClock::new(0, 0.1, 0.5, 1.0);
        // error = drift * t + latency = 1.5 at t=10
        let corrected = clock.corrected_time(10.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0);
        assert!((corrected - 10.0).abs() < 1e-10);
    }
}
