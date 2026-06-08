#[cfg(test)]
mod tests {
    use fibration_timing::bundle::FiberBundle;
    use fibration_timing::connection::Connection;
    use fibration_timing::drift::DriftMonitor;
    use fibration_timing::error::FiberError;
    use fibration_timing::holonomy::{
        detect_loop_holonomy, detect_triangle_holonomies, HolonomyReport,
    };
    use fibration_timing::schedule::{distribute_agents, AgentSchedule, ScheduleBuilder};

    // ---- FiberBundle tests ----

    #[test]
    fn bundle_creation_basic() {
        let b = FiberBundle::new(vec![0.0, 1.0, 2.0], 3).unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(b.fiber_dim, 3);
    }

    #[test]
    fn bundle_empty_base_error() {
        let e = FiberBundle::new(vec![], 2);
        assert!(matches!(e, Err(FiberError::EmptyBaseSpace)));
    }

    #[test]
    fn bundle_zero_fiber_error() {
        let e = FiberBundle::new(vec![0.0], 0);
        assert!(matches!(e, Err(FiberError::InvalidFiberDimension(0))));
    }

    #[test]
    fn bundle_projection() {
        let b = FiberBundle::new(vec![0.0, 1.5, 3.0], 2).unwrap();
        assert_eq!(b.projection(0).unwrap(), 0.0);
        assert_eq!(b.projection(1).unwrap(), 1.5);
        assert_eq!(b.projection(2).unwrap(), 3.0);
    }

    #[test]
    fn bundle_projection_oob() {
        let b = FiberBundle::new(vec![0.0, 1.0], 1).unwrap();
        assert!(b.projection(5).is_err());
    }

    #[test]
    fn bundle_nearest_index() {
        let b = FiberBundle::new(vec![0.0, 1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(b.nearest_index(0.4), 0);
        assert_eq!(b.nearest_index(0.6), 1);
        assert_eq!(b.nearest_index(2.9), 3);
    }

    // ---- Connection tests ----

    #[test]
    fn flat_connection_identity_transport() {
        let c = Connection::flat(2, 3);
        let v = vec![3.0, 4.0];
        let transported = c.transport_vector(&v, 0, 2).unwrap();
        assert!((transported[0] - 3.0).abs() < 1e-9);
        assert!((transported[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn flat_connection_round_trip() {
        let c = Connection::flat(3, 5);
        let v = vec![1.0, 2.0, 3.0];
        let forward = c.transport_vector(&v, 0, 4).unwrap();
        let back = c.transport_vector(&forward, 4, 0).unwrap();
        for i in 0..3 {
            assert!((back[i] - v[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn perturbed_connection_not_identity() {
        let c = Connection::perturbed(2, 3, 42);
        let v = vec![1.0, 0.0];
        let t = c.transport_vector(&v, 0, 2).unwrap();
        // With perturbation, result should differ from identity transport
        // (not guaranteed for all seeds, but very likely for seed 42)
        let diff = (t[0] - 1.0).abs() + (t[1] - 0.0).abs();
        assert!(diff > 0.001, "perturbed connection should differ from identity");
    }

    #[test]
    fn holonomy_flat_is_zero() {
        let c = Connection::flat(2, 5);
        let hol = c.holonomy_matrix(&vec![0, 1, 2, 0]).unwrap();
        let drift = frobenius_to_identity(&hol);
        assert!(drift < 1e-9, "flat connection should have zero holonomy");
    }

    #[test]
    fn holonomy_loop_not_closed_error() {
        let c = Connection::flat(2, 3);
        let result = c.holonomy_matrix(&vec![0, 1, 2]);
        assert!(matches!(result, Err(FiberError::LoopNotClosed { .. })));
    }

    #[test]
    fn holonomy_loop_too_short() {
        let c = Connection::flat(2, 3);
        let result = c.holonomy_matrix(&vec![0]);
        assert!(matches!(result, Err(FiberError::LoopTooShort(1))));
    }

    // ---- Holonomy detection tests ----

    #[test]
    fn detect_triangles_flat_no_reports() {
        let c = Connection::flat(2, 5);
        let reports = detect_triangle_holonomies(
            &c.transport,
            6,
            0.01,
            &["a".to_string()],
        );
        assert!(reports.is_empty());
    }

    #[test]
    fn detect_loop_holonomy_below_threshold() {
        let hol = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let report = detect_loop_holonomy(
            &hol,
            vec![0, 1, 0],
            vec!["agent".to_string()],
            0.1,
        );
        assert!(report.is_none());
    }

    #[test]
    fn holonomy_report_is_negligible() {
        let r = HolonomyReport {
            loop_path: vec![0, 1, 0],
            drift: 0.001,
            agent_ids: vec![],
        };
        assert!(r.is_negligible(0.01));
        assert!(!r.is_negligible(0.0001));
    }

    // ---- ScheduleBuilder tests ----

    #[test]
    fn schedule_two_agents() {
        let bundle = FiberBundle::new(vec![0.0, 1.0, 2.0], 2).unwrap();
        let schedules = ScheduleBuilder::new(bundle)
            .add_agent("a1", vec![1.0, 0.0], 1)
            .unwrap()
            .add_agent("a2", vec![0.0, 1.0], 2)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(schedules.len(), 2);
        // Higher priority first
        assert_eq!(schedules[0].agent_id, "a2");
        assert_eq!(schedules[1].agent_id, "a1");
    }

    #[test]
    fn schedule_no_agents_error() {
        let bundle = FiberBundle::new(vec![0.0, 1.0], 1).unwrap();
        let result = ScheduleBuilder::new(bundle).build();
        assert!(matches!(result, Err(FiberError::NoAgents)));
    }

    #[test]
    fn schedule_wrong_state_dim() {
        let bundle = FiberBundle::new(vec![0.0, 1.0], 2).unwrap();
        let result = ScheduleBuilder::new(bundle).add_agent("x", vec![1.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn schedule_time_slots_match_base() {
        let bundle = FiberBundle::new(vec![0.0, 0.5, 1.0, 1.5], 1).unwrap();
        let schedules = ScheduleBuilder::new(bundle)
            .add_agent("a", vec![1.0], 1)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(schedules[0].time_slots, vec![0.0, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn schedule_flat_transport_preserves_state() {
        let bundle = FiberBundle::new(vec![0.0, 1.0, 2.0], 2).unwrap();
        let schedules = ScheduleBuilder::new(bundle)
            .add_agent("a", vec![3.0, 4.0], 1)
            .unwrap()
            .build()
            .unwrap();
        for state in &schedules[0].fiber_states {
            assert!((state[0] - 3.0).abs() < 1e-9);
            assert!((state[1] - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn schedule_with_holonomy() {
        let bundle = FiberBundle::new(vec![0.0, 1.0, 2.0, 3.0], 2).unwrap();
        let (schedules, reports) = ScheduleBuilder::new(bundle)
            .add_agent("a", vec![1.0, 0.0], 1)
            .unwrap()
            .build_with_holonomy(0.01)
            .unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(reports.is_empty()); // flat connection
    }

    // ---- distribute_agents tests ----

    #[test]
    fn distribute_basic() {
        let result = distribute_agents(3, 6);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (0, vec![0, 3]));
        assert_eq!(result[1], (1, vec![1, 4]));
        assert_eq!(result[2], (2, vec![2, 5]));
    }

    #[test]
    fn distribute_empty() {
        assert!(distribute_agents(0, 5).is_empty());
        assert!(distribute_agents(3, 0).is_empty());
    }

    // ---- DriftMonitor tests ----

    #[test]
    fn drift_no_violations() {
        let sched = AgentSchedule {
            agent_id: "a".into(),
            time_slots: vec![0.0, 1.0, 2.0],
            fiber_states: vec![vec![0.0], vec![0.0], vec![0.0]],
            priority: 1,
        };
        let mut mon = DriftMonitor::new(sched, 0.5);
        mon.record(0, 0.1).unwrap();
        mon.record(1, 1.2).unwrap();
        mon.record(2, 1.8).unwrap();
        assert!(mon.violations().is_empty());
        assert!(mon.check().is_ok());
    }

    #[test]
    fn drift_with_violations() {
        let sched = AgentSchedule {
            agent_id: "a".into(),
            time_slots: vec![0.0, 1.0, 2.0],
            fiber_states: vec![vec![0.0], vec![0.0], vec![0.0]],
            priority: 1,
        };
        let mut mon = DriftMonitor::new(sched, 0.1);
        mon.record(0, 0.05).unwrap();
        mon.record(1, 1.5).unwrap(); // 0.5 drift
        mon.record(2, 2.02).unwrap();
        assert_eq!(mon.violations(), vec![1]);
        assert!(mon.check().is_err());
    }

    #[test]
    fn drift_max_and_mean() {
        let sched = AgentSchedule {
            agent_id: "a".into(),
            time_slots: vec![0.0, 1.0, 2.0],
            fiber_states: vec![vec![], vec![], vec![]],
            priority: 1,
        };
        let mut mon = DriftMonitor::new(sched, 1.0);
        mon.record(0, 0.1).unwrap();
        mon.record(1, 1.3).unwrap();
        mon.record(2, 1.7).unwrap();
        assert!((mon.max_drift() - 0.3).abs() < 1e-9);
        assert!((mon.mean_drift() - 0.7 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn drift_record_oob() {
        let sched = AgentSchedule {
            agent_id: "a".into(),
            time_slots: vec![0.0],
            fiber_states: vec![vec![]],
            priority: 1,
        };
        let mut mon = DriftMonitor::new(sched, 1.0);
        assert!(mon.record(5, 0.0).is_err());
    }

    #[test]
    fn drift_slot_count() {
        let sched = AgentSchedule {
            agent_id: "a".into(),
            time_slots: vec![0.0, 1.0, 2.0, 3.0],
            fiber_states: vec![vec![], vec![], vec![], vec![]],
            priority: 1,
        };
        let mon = DriftMonitor::new(sched, 1.0);
        assert_eq!(mon.slot_count(), 4);
    }

    // ---- Integration: perturbed bundle + drift ----

    #[test]
    fn perturbed_bundle_schedule_drift() {
        let conn = Connection::perturbed(2, 4, 123);
        let bundle = FiberBundle::with_connection(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            2,
            conn,
        )
        .unwrap();
        let schedules = ScheduleBuilder::new(bundle)
            .add_agent("a", vec![1.0, 0.0], 1)
            .unwrap()
            .build()
            .unwrap();
        let s = &schedules[0];
        // Fiber states should change with perturbed connection
        assert_ne!(s.fiber_states[0], s.fiber_states[4]);
    }

    fn frobenius_to_identity(mat: &[Vec<f64>]) -> f64 {
        let n = mat.len();
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = mat[i][j] - expected;
                sum += diff * diff;
            }
        }
        sum.sqrt()
    }
}
