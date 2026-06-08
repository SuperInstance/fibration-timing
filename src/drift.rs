use serde::{Deserialize, Serialize};

use crate::error::FiberError;
use crate::schedule::AgentSchedule;

/// Monitors timing drift of an agent against its scheduled time slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMonitor {
    /// The expected schedule.
    pub schedule: AgentSchedule,
    /// Actual execution times recorded for each slot.
    pub actual_times: Vec<f64>,
    /// Maximum allowed drift before flagging a violation.
    pub tolerance: f64,
}

impl DriftMonitor {
    /// Create a new drift monitor for the given schedule.
    pub fn new(schedule: AgentSchedule, tolerance: f64) -> Self {
        let n = schedule.time_slots.len();
        Self {
            schedule,
            actual_times: vec![0.0; n],
            tolerance,
        }
    }

    /// Record an actual execution time for the given slot index.
    pub fn record(&mut self, slot_idx: usize, actual_time: f64) -> Result<(), FiberError> {
        if slot_idx >= self.schedule.time_slots.len() {
            return Err(FiberError::IndexOutOfBounds {
                index: slot_idx,
                len: self.schedule.time_slots.len(),
            });
        }
        self.actual_times[slot_idx] = actual_time;
        Ok(())
    }

    /// Compute the drift at each recorded slot.
    ///
    /// Returns a vec of (scheduled, actual, drift) tuples.
    pub fn compute_drifts(&self) -> Vec<(f64, f64, f64)> {
        self.schedule
            .time_slots
            .iter()
            .zip(self.actual_times.iter())
            .map(|(scheduled, actual)| {
                let drift = (actual - scheduled).abs();
                (*scheduled, *actual, drift)
            })
            .collect()
    }

    /// Return indices of slots where drift exceeds tolerance.
    pub fn violations(&self) -> Vec<usize> {
        self.compute_drifts()
            .iter()
            .enumerate()
            .filter_map(|(i, &(_, _, drift))| {
                if drift > self.tolerance {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Maximum drift across all recorded slots.
    pub fn max_drift(&self) -> f64 {
        self.compute_drifts()
            .iter()
            .map(|&(_, _, d)| d)
            .fold(0.0_f64, f64::max)
    }

    /// Mean drift across all recorded slots.
    pub fn mean_drift(&self) -> f64 {
        let drifts = self.compute_drifts();
        if drifts.is_empty() {
            return 0.0;
        }
        drifts.iter().map(|&(_, _, d)| d).sum::<f64>() / drifts.len() as f64
    }

    /// Check all slots and return an error for the first violation found.
    pub fn check(&self) -> Result<(), FiberError> {
        for (i, &(_, _, drift)) in self.compute_drifts().iter().enumerate() {
            if drift > self.tolerance {
                return Err(FiberError::DriftExceeded {
                    drift,
                    tolerance: self.tolerance,
                    agent_id: self.schedule.agent_id.clone(),
                });
            }
            let _ = i;
        }
        Ok(())
    }

    /// Number of slots being monitored.
    pub fn slot_count(&self) -> usize {
        self.schedule.time_slots.len()
    }
}
