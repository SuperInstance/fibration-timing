use thiserror::Error;

/// Errors arising from fiber-bundle timing operations.
#[derive(Debug, Error)]
pub enum FiberError {
    #[error("empty base space: timeline must contain at least one point")]
    EmptyBaseSpace,

    #[error("fiber dimension must be ≥ 1, got {0}")]
    InvalidFiberDimension(usize),

    #[error("connection transport matrix size mismatch: expected {expected}×{expected}, got {got}×{got}")]
    TransportMatrixSizeMismatch { expected: usize, got: usize },

    #[error("index {index} out of bounds for base space of length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("loop path must contain ≥ 2 points, got {0}")]
    LoopTooShort(usize),

    #[error("loop path is not closed: start {start} ≠ end {end}")]
    LoopNotClosed { start: usize, end: usize },

    #[error("no agents to schedule")]
    NoAgents,

    #[error("drift {drift} exceeds tolerance {tolerance} for agent {agent_id}")]
    DriftExceeded {
        drift: f64,
        tolerance: f64,
        agent_id: String,
    },
}
