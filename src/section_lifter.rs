use crate::bundle::{AgentClock, TimingBundle};
use serde::{Deserialize, Serialize};

/// Lifts base-level schedules to individual agent timelines (sections of the bundle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionLifter {
    pub bundle: TimingBundle,
}

impl SectionLifter {
    pub fn new(bundle: TimingBundle) -> Self {
        Self { bundle }
    }

    /// Lift a base time to all agent local times (a section of the bundle).
    pub fn lift(&self, base_time: f64) -> Vec<f64> {
        self.bundle
            .clocks
            .iter()
            .map(|c| c.local_time(base_time))
            .collect()
    }

    /// Lift a base schedule (list of base times) to all agent timelines.
    pub fn lift_schedule(&self, base_times: &[f64]) -> Vec<Vec<f64>> {
        base_times.iter().map(|t| self.lift(*t)).collect()
    }

    /// Lift with PID correction applied.
    pub fn lift_corrected(
        &self,
        base_time: f64,
        kp: f64,
        ki: f64,
        kd: f64,
        errors: &[f64],
        integrals: &[f64],
        derivatives: &[f64],
    ) -> Vec<f64> {
        self.bundle
            .clocks
            .iter()
            .zip(errors.iter())
            .zip(integrals.iter())
            .zip(derivatives.iter())
            .map(|(((c, &e), &i), &d)| c.corrected_time(base_time, kp, ki, kd, e, i, d))
            .collect()
    }

    /// Compute the desynchronization between agents at a given base time.
    /// Returns max pairwise difference in local times.
    pub fn desynchronization_at(&self, base_time: f64) -> f64 {
        let times = self.lift(base_time);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }

    /// Compute agent timeline from base timeline, returning the per-agent trajectory.
    pub fn agent_timeline(&self, agent_id: usize, base_times: &[f64]) -> Vec<f64> {
        base_times
            .iter()
            .map(|t| self.bundle.clocks[agent_id].local_time(*t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_lift_single_agent() {
        let clock = AgentClock::new(0, 0.0, 0.0, 1.0);
        let bundle = TimingBundle::with_clocks(vec![clock], 1, 1);
        let lifter = SectionLifter::new(bundle);
        let times = lifter.lift(5.0);
        assert_eq!(times.len(), 1);
        assert!((times[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn section_lift_two_agents() {
        let clocks = vec![
            AgentClock::new(0, 0.0, 0.0, 1.0),
            AgentClock::new(1, 0.1, 0.5, 1.0),
        ];
        let bundle = TimingBundle::with_clocks(clocks, 1, 1);
        let lifter = SectionLifter::new(bundle);
        let times = lifter.lift(10.0);
        assert!((times[0] - 10.0).abs() < 1e-10);
        assert!((times[1] - 11.5).abs() < 1e-10);
    }

    #[test]
    fn desynchronization_two_agents() {
        let clocks = vec![
            AgentClock::new(0, 0.0, 0.0, 1.0),
            AgentClock::new(1, 0.1, 0.5, 1.0),
        ];
        let bundle = TimingBundle::with_clocks(clocks, 1, 1);
        let lifter = SectionLifter::new(bundle);
        // At t=10: agent0=10, agent1=11.5, desync=1.5
        assert!((lifter.desynchronization_at(10.0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn lift_schedule_produces_correct_shape() {
        let clocks = vec![
            AgentClock::new(0, 0.0, 0.0, 1.0),
            AgentClock::new(1, 0.0, 0.0, 1.0),
        ];
        let bundle = TimingBundle::with_clocks(clocks, 1, 1);
        let lifter = SectionLifter::new(bundle);
        let schedule = lifter.lift_schedule(&[1.0, 2.0, 3.0]);
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0].len(), 2);
    }

    #[test]
    fn agent_timeline_matches_clock() {
        let clock = AgentClock::new(0, 0.05, 1.0, 2.0);
        let bundle = TimingBundle::with_clocks(vec![clock], 1, 1);
        let lifter = SectionLifter::new(bundle);
        let timeline = lifter.agent_timeline(0, &[0.0, 1.0, 2.0]);
        assert!((timeline[0] - 1.0).abs() < 1e-10); // 0*1.05 + 1.0
        assert!((timeline[1] - 2.05).abs() < 1e-10);
        assert!((timeline[2] - 3.1).abs() < 1e-10);
    }
}
