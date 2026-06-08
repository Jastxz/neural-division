---
inclusion: fileMatch
fileMatchPattern: "test/**/*.jl"
---

# Testing — DivisionNeuronal

## Framework
- Julia `Test` stdlib for unit tests
- PropCheck for property-based testing

## Run
- `julia --project=. -e 'using Pkg; Pkg.test()'`
- Or: `julia --project=. test/runtests.jl`

## Conventions
- Test files in `test/` directory
- One `runtests.jl` entry point that includes sub-test files
- Property tests for numerical invariants (symmetry, bounds, convergence)
- Use `@testset` blocks for grouping
- Seed RNG explicitly: `Random.seed!(42)` for reproducibility
