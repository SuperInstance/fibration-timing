//! # fibration-timing
//!
//! Fiber bundle timing for concurrent agent scheduling.
//!
//! This crate models concurrent agent scheduling as a fiber bundle
//! **(E, B, π, F)** where:
//!
//! - **Base space B** = a timeline (ordered set of time points).
//! - **Fiber F** = the agent state space (a vector space of dimension `fiber_dim`).
//! - **Total space E** = all possible (time, state) pairs.
//! - **Projection π** maps each (time, state) pair to its time coordinate.
//!
//! A **connection** on the bundle defines parallel transport of agent state
//! along the timeline. **Holonomy** (the failure of parallel transport around
//! a closed loop to return to the identity) measures scheduling drift — the
//! accumulated error when agents are transported through their scheduled time
//! slots.
//!
//! ## Core concepts
//!
//! | Concept | Scheduling meaning |
//! |---------|-------------------|
//! | Base space | Timeline of execution slots |
//! | Fiber | Agent state vector |
//! | Connection | Time-evolution rule |
//! | Curvature | Non-commutativity of scheduling order |
//! | Holonomy | Accumulated scheduling drift |
//!
//! ## Example
//!
//! ```
//! use fibration_timing::bundle::FiberBundle;
//! use fibration_timing::schedule::ScheduleBuilder;
//!
//! let bundle = FiberBundle::new(vec![0.0, 1.0, 2.0, 3.0], 2).unwrap();
//! let schedules = ScheduleBuilder::new(bundle)
//!     .add_agent("agent-0", vec![1.0, 0.0], 1).unwrap()
//!     .add_agent("agent-1", vec![0.0, 1.0], 2).unwrap()
//!     .build()
//!     .unwrap();
//! assert_eq!(schedules.len(), 2);
//! ```

pub mod bundle;
pub mod connection;
pub mod drift;
pub mod error;
pub mod holonomy;
pub mod schedule;

pub use bundle::FiberBundle;
pub use connection::Connection;
pub use drift::DriftMonitor;
pub use error::FiberError;
pub use holonomy::HolonomyReport;
pub use schedule::{AgentSchedule, ScheduleBuilder};
