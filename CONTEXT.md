# Project Context — Agent-AI4EDA

## Current State (as of 2026-05-19)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/analog-agent |
| Commits | 65 |
| Stage | Paper submission preparation |
| 60-day plan | COMPLETED (all workstreams) |
| Vertical slices | 4 frozen (ota2, folded_cascode, ldo, bandgap) |
| Baselines | 5 (random, BayesOpt, CMA-ES, RL, no-world-model) |
| Test coverage | Unit + integration + regression |
| Python | 3.11+ |
| Simulator | ngspice 46 (local), Xyce (planned), Spectre (planned) |

## Architecture (Six Layers)
1. Specification Understanding — NL → structured spec
2. Task Formalization — spec → design space + constraints
3. World Model — surrogate prediction (GNN/MLP/XGB)
4. Planning & Optimization — BO/CMA-ES/MPC search
5. Simulation & Verification — real SPICE truth model
6. Memory & Reflection — cross-episode learning

## Key Decisions
- ngspice is the truth model (not mocked)
- World model guides search but SPICE verifies
- Four circuits are frozen benchmarks (no cherry-picking)
- Paper targets top-tier EDA venue (DAC/ICCAD/DATE class)

## What's Next
- [ ] External PDK integration (sky130 real fab-ready runs)
- [ ] Spectre backend for industry compatibility
- [ ] Final manuscript assembly with all figures/tables
- [ ] Submission package (code + data + reproducibility)
