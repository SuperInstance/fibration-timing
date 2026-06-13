# fibration-timing

A Rust library modeling temporal coordination in multi-agent dialogue as a **fiber bundle** over a base timeline.

## Mathematical Framework

In a multi-agent dialogue system, each agent operates on its own local clock. The shared conversation timeline is the **base manifold**, and each agent's temporal state is a **fiber**. The full space of agent timings forms a **principal fiber bundle**:

$$\pi: E \to B$$

where $E$ is the total space of all agent timelines, $B$ is the shared base timeline, and the fiber over each point is the product of agent clock states.

### Key Concepts

| Bundle Concept | Dialogue Interpretation |
|---|---|
| **Base manifold** $B$ | Shared conversation timeline |
| **Fiber** $F$ | Each agent's local temporal state (clock) |
| **Connection form** $\omega$ | Rules for transporting timing between agents |
| **Curvature** $\Omega = d\omega + \omega \wedge \omega$ | Rate of desynchronization per dialogue exchange |
| **Holonomy** | Net timing drift around a closed dialogue cycle |
| **Section** | A specific assignment of agent times to each base time |
| **Gauge choice** | Choice of canonical synchronization frame |

## Modules

- **`bundle`** — Principal timing bundle: `TimingBundle` with agent `AgentClock` fibers carrying drift, latency, and rhythm.
- **`connection`** — Ehresmann connection forms (`ConnectionForm`) for clock transport between agents. Horizontal lift of base curves via parallel transport.
- **`holonomy`** — `HolonomyGroup` computation around dialogue cycles. Net timing drift around closed loops.
- **`curvature`** — Curvature 2-form (`CurvatureForm`): rate of desynchronization per unit dialogue exchange.
- **`section_lifter`** — `SectionLifter` maps base-level schedules to individual agent timelines.
- **`gauge_fix`** — `GaugeFixer` for choosing canonical synchronization frames (Lorenz-like gauge).

## Core Types

```rust
struct TimingBundle {
    base_dim: usize,   // Shared timeline dimensionality
    fiber_dim: usize,  // Agent state dimensionality
    agents: usize,     // Number of agents
    clocks: Vec<AgentClock>,
}

struct AgentClock {
    id: usize,
    drift_rate: f64,   // Fractional clock drift per unit base time
    latency: f64,      // Fixed offset
    rhythm: f64,       // Natural response cadence period
}

struct ConnectionForm {
    components: Vec<Vec<f64>>,  // Matrix components of ω
}

struct CurvatureForm {
    components: Vec<Vec<Vec<f64>>>,  // Ω = dω + ω∧ω
}

struct HolonomyGroup {
    generators: Vec<Vec<f64>>,  // Holonomy matrices
    drift: f64,                 // Net accumulated drift
}
```

## Usage

```rust
use fibration_timing::{TimingBundle, AgentClock, ConnectionForm, SectionLifter, HolonomyGroup, GaugeFixer};

// Create a bundle with two agents
let clocks = vec![
    AgentClock::new(0, 0.0, 0.0, 1.0),   // Perfect clock
    AgentClock::new(1, 0.1, 0.5, 2.0),   // Drifting clock
];
let bundle = TimingBundle::with_clocks(clocks, 1, 1);

// Lift base schedule to agent timelines
let lifter = SectionLifter::new(bundle);
let agent_times = lifter.lift(10.0);
// agent_times[0] = 10.0, agent_times[1] = 11.5

// Compute curvature from a connection
let omega = ConnectionForm::new(vec![
    vec![0.0, 0.1],
    vec![-0.1, 0.0],
]);
let curvature = omega.curvature();
// Curvature measures desynchronization rate

// Compute holonomy around a dialogue cycle
let h = HolonomyGroup::rectangular_cycle(&omega, 2.0, 3.0, 100);
// h.drift = net timing misalignment after one cycle

// Gauge fix (Lorenz-like frame)
let fixer = GaugeFixer::lorenz(2);
let canonical = fixer.fix(&omega);
```

## Properties Verified by Tests (40 tests)

1. **Single agent (trivial bundle)** → curvature = 0, holonomy = identity
2. **Two agents with constant drift** → constant curvature (for non-nilpotent connections)
3. **Holonomy around dialogue cycle** = drift × cycle area
4. **PID-corrected agents** → error decays exponentially
5. **Structure equation**: $\Omega = d\omega + \omega \wedge \omega$
6. **Parallel transport** along paths preserves fiber metric (for so(n) connections)
7. **Gauge transformation** preserves curvature norm (invariance)
8. **Flatness criterion**: zero curvature ↔ globally synchronizable
9. **Section lifting** produces correct agent timelines
10. **Gaussian elimination** from scratch (no external deps)

## Design Decisions

- **No external math crates**: All linear algebra (Gaussian elimination, matrix exp, etc.) implemented from scratch.
- **Serde on all public types**: Full serialization support.
- **Edition 2024**: Modern Rust idioms.
- **Numerical methods**: Matrix exponential via Taylor series (scaling-and-squaring for small matrices).

## License

MIT
