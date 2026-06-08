---
inclusion: always
---

# Architecture — neural-division (DivisionNeuronal)

## Stack
- **Language:** Julia 1.9+
- **Build:** Pkg (Project.toml / Manifest.toml)
- **Key deps:** Plots, UnicodePlots, JLD2, CodecZlib
- **GPU:** Optional CUDA/AMDGPU via weak dependencies

## Module Boundaries (from graphify: 218 nodes, 264 edges, 47 communities)
- God nodes: `main()` (9 edges), `tabla_verdad()` (7), `exp15()` (6), `exp9_sinteticos()` (6)
- `src/` — Main library code (DivisionNeuronal module)
  - Neural network architecture with division-based topology
  - Training and inference pipelines
  - Serialization via JLD2
  - `tabla_verdad()` — Truth table generation for verification
  - Experiment functions (`exp15()`, `exp9_sinteticos()`) for research validation
- `test/` — Unit tests with PropCheck (property-based testing)
- `benchmarks/` — Performance benchmarks
- `csv/` — Data files
- `articulo/` — Related paper/article

## Dependency Rules
- Core logic in `src/` must not depend on plotting — keep Plots.jl in scripts/notebooks
- GPU extensions via Julia's extension mechanism (weak deps)
- Data serialization via JLD2 — compressed with CodecZlib
- Random seeding explicit for reproducibility

## Design Principles
- Pure numerical Julia — leverage multiple dispatch
- GPU support opt-in, not required
- Property-based tests via PropCheck for correctness guarantees
